from sqlmodel import SQLModel, Session, create_engine
import os
from langchain.tools import tool
from sqlmodel import Session
from sqlalchemy import text
from textwrap import shorten
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ['PG_DSN']="postgresql://postgres:postgres@localhost:5432/langgraph_db"
os.environ['GOOGLE_API_KEY']=""

def get_engine():
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN environment variable not set")
    return create_engine(dsn, echo=False)

engine = get_engine()


@tool
def run_sql_query(sql: str, max_rows: int = 20) -> str:
    """
    Execute a read-only SQL SELECT statement using SQLModel and return results
    as a markdown table.

    Only SELECT queries are allowed (safety guard).
    """
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("This tool only supports SELECT queries.")

    with Session(engine) as session:
        result = session.exec(text(sql))
        rows = result.fetchall()
        if not rows:
            return "Query executed successfully, but returned 0 rows."

        col_names = result.keys()
        rows = rows[:max_rows]

    # Format as Markdown
    header = "| " + " | ".join(col_names) + " |"
    sep = "| " + " | ".join("---" for _ in col_names) + " |"

    body = []
    for r in rows:
        body.append(
            "| "
            + " | ".join(shorten(str(v), width=80, placeholder="…") for v in r)
            + " |"
        )

    table = "\n".join([header, sep] + body)
    if len(body) == max_rows:
        table += f"\n\n_NOTE: truncated to {max_rows} rows._"

    return table

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1,
)

agent = create_agent(
    model=llm,
    tools=[run_sql_query],
)

result = agent.invoke({
    "messages": [
        HumanMessage(content="Show me how many employees from France.")
        #HumanMessage(content="Show me the last 10 employees ordered by salary desc.")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()

"""
uv --project uv_env/ run python week_05/05_tools/query_database.py 
================================ Human Message =================================

Show me how many employees from France.
================================== Ai Message ==================================
Tool Calls:
  run_sql_query (e811cbe7-992a-4f88-b167-3e33bc9aa15f)
 Call ID: e811cbe7-992a-4f88-b167-3e33bc9aa15f
  Args:
    sql: SELECT count(*) FROM employees WHERE country = 'France'
================================= Tool Message =================================
Name: run_sql_query

| count |
| --- |
| 10 |
================================== Ai Message ==================================

There are 10 employees from France.
"""