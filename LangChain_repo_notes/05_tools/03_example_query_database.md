In this example we will query database as tool and get the answer of "how many employees from France".

## ✔️ Assumptions

* You want:
  **Option A — run_sql_query(sql: str) → returns query result as markdown**
* DB operations must use **SQLModel** + **SQLAlchemy 2.0 engine**
* Compatible with **langchain>=1.0.5**
* LLM = **Gemini 2.5 Flash** (`ChatGoogleGenerativeAI`)

## Prepare database
```
docker run -d --rm \
--name postgres-langgraph \
-e POSTGRES_USER=postgres \
-e POSTGRES_PASSWORD=postgres \
-e POSTGRES_DB=langgraph_db \
-p 5432:5432 \
postgres:16 

# Connect postgres
docker exec -it postgres-langgraph psql -U postgres -d langgraph_db
```

- Create table
```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    full_name        VARCHAR(100) NOT NULL,
    department       VARCHAR(50)  NOT NULL,
    salary           INTEGER      NOT NULL CHECK (salary > 0),
    country          VARCHAR(50)  NOT NULL,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW()
);
```
- Insert records
```sql
BEGIN;

INSERT INTO employees (full_name, department, salary, country)
VALUES
 ('John Smith', 'Engineering', 124000, 'USA'),
  ('Emily Davis', 'Marketing', 78000, 'Canada'),
  ('Michael Johnson', 'Finance', 96000, 'UK'),
  ('Sarah Wilson', 'HR', 72000, 'Germany'),
  ('David Brown', 'Engineering', 135000, 'USA'),
  ('Laura Miller', 'Sales', 89000, 'France'),
  ('Daniel Garcia', 'Engineering', 145000, 'Spain'),
  ('Sophia Martinez', 'Finance', 101000, 'USA'),
  ('James Anderson', 'Marketing', 81000, 'Netherlands'),
  ('Olivia Thomas', 'HR', 67000, 'Turkey'),
  ('Benjamin Lee', 'Engineering', 155000, 'South Korea'),
  ('Isabella White', 'Sales', 92000, 'USA'),
  ('Alexander Harris', 'Engineering', 130000, 'Germany'),
  ('Mia Clark', 'Finance', 98000, 'Canada'),
  ('William Lewis', 'HR', 69000, 'USA'),
  ('Evelyn Young', 'Marketing', 83000, 'UK'),
  ('Henry Walker', 'Engineering', 148000, 'USA'),
  ('Ava Hall', 'Sales', 91000, 'France'),
  ('Matthew Allen', 'Engineering', 158000, 'USA'),
  ('Grace King', 'Finance', 102000, 'Spain'),
  ('Samuel Wright', 'HR', 74000, 'Germany'),
  ('Chloe Scott', 'Marketing', 85000, 'Canada'),
  ('Joseph Green', 'Engineering', 142000, 'USA'),
  ('Victoria Adams', 'Sales', 93000, 'Turkey'),
  ('Andrew Baker', 'Finance', 97000, 'UK'),
  ('Ella Nelson', 'HR', 70000, 'USA'),
  ('Christopher Carter', 'Engineering', 152000, 'Netherlands'),
  ('Scarlett Mitchell', 'Marketing', 82000, 'France'),
  ('Jack Perez', 'Engineering', 149000, 'Spain'),
  ('Luna Roberts', 'Sales', 94000, 'USA'),
  ('Owen Turner', 'Finance', 104000, 'Germany'),
  ('Harper Phillips', 'HR', 76000, 'Canada'),
  ('Gabriel Campbell', 'Engineering', 160000, 'USA'),
  ('Aria Parker', 'Marketing', 88000, 'Turkey'),
  ('Logan Evans', 'Engineering', 157000, 'USA'),
  ('Penelope Edwards', 'Sales', 92500, 'Spain'),
  ('Sebastian Collins', 'Finance', 112000, 'UK'),
  ('Nora Stewart', 'HR', 72000, 'France'),
  ('Elijah Sanchez', 'Engineering', 143000, 'USA'),
  ('Lily Morris', 'Marketing', 86000, 'Canada'),
  ('Mason Rogers', 'Engineering', 154000, 'Germany'),
  ('Zoe Reed', 'Sales', 94500, 'USA'),
  ('Jacob Cook', 'Finance', 108000, 'Spain'),
  ('Stella Morgan', 'HR', 73000, 'Turkey'),
  ('Luke Bell', 'Engineering', 151000, 'USA'),
  ('Hannah Murphy', 'Marketing', 82000, 'France'),
  ('Jayden Bailey', 'Engineering', 147000, 'UK'),
  ('Natalie Rivera', 'Sales', 93500, 'Canada'),
  ('David Cooper', 'Finance', 110000, 'USA'),
  ('Riley Richardson', 'HR', 75000, 'Netherlands'),
  ('Ethan Cox', 'Engineering', 140000, 'USA'),
  ('Leah Howard', 'Marketing', 87000, 'Germany'),
  ('Caleb Ward', 'Engineering', 153000, 'USA'),
  ('Violet Torres', 'Sales', 95000, 'Spain'),
  ('Jonathan Peterson', 'Finance', 115000, 'USA'),
  ('Audrey Gray', 'HR', 74000, 'France'),
  ('Isaac Ramirez', 'Engineering', 146000, 'Canada'),
  ('Camila James', 'Marketing', 89000, 'UK'),
  ('Wyatt Watson', 'Engineering', 150000, 'USA'),
  ('Layla Brooks', 'Sales', 97000, 'Germany'),
  ('Dylan Kelly', 'Finance', 107000, 'Spain'),
  ('Paisley Sanders', 'HR', 76000, 'USA'),
  ('Nathan Price', 'Engineering', 159000, 'Netherlands'),
  ('Ellie Bennett', 'Marketing', 88000, 'Turkey'),
  ('Ryan Wood', 'Engineering', 144000, 'France'),
  ('Brooklyn Barnes', 'Sales', 96000, 'USA'),
  ('Asher Ross', 'Finance', 109000, 'Canada'),
  ('Savannah Henderson', 'HR', 74500, 'USA'),
  ('Julian Coleman', 'Engineering', 156000, 'UK'),
  ('Avery Jenkins', 'Marketing', 87000, 'Spain'),
  ('Hunter Perry', 'Engineering', 148000, 'USA'),
  ('Claire Powell', 'Sales', 92000, 'Germany'),
  ('Christian Long', 'Finance', 103000, 'Turkey'),
  ('Skylar Patterson', 'HR', 77000, 'France'),
  ('Cameron Hughes', 'Engineering', 141000, 'Canada'),
  ('Lucy Flores', 'Marketing', 84000, 'USA'),
  ('Adam Washington', 'Engineering', 152000, 'Spain'),
  ('Sadie Butler', 'Sales', 93500, 'UK'),
  ('Connor Simmons', 'Finance', 111000, 'USA'),
  ('Madelyn Foster', 'HR', 72000, 'Netherlands'),
  ('Brayden Gonzales', 'Engineering', 145000, 'Germany'),
  ('Hailey Bryant', 'Marketing', 83000, 'Canada'),
  ('Bentley Alexander', 'Engineering', 160000, 'USA'),
  ('Sofia Russell', 'Sales', 94000, 'Turkey'),
  ('Eli Griffin', 'Finance', 106000, 'France'),
  ('Aubrey Diaz', 'HR', 75000, 'Spain'),
  ('Carson Hayes', 'Engineering', 149000, 'USA'),
  ('Ariana Myers', 'Marketing', 88000, 'Germany'),
  ('Dominic Ford', 'Engineering', 155000, 'Canada'),
  ('Anna Hamilton', 'Sales', 92000, 'UK'),
  ('Jaxon Graham', 'Finance', 108500, 'USA'),
  ('Piper Sullivan', 'HR', 76000, 'France');
COMMIT;
```
---

# 1. Install Requirements

```bash
pip install -U "sqlmodel>=0.0.14" "psycopg[binary]" "langchain>=1.0.5" langchain-google-genai
```

`sqlmodel` uses SQLAlchemy under the hood and works perfectly with PostgreSQL.

---

# 2. SQLModel Engine + Session Helper

```python
from sqlmodel import SQLModel, Session, create_engine
import os

os.environ['PG_DSN']="postgresql://postgres:postgres@localhost:5432/langgraph_db"
os.environ['GOOGLE_API_KEY']=""

def get_engine():
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN environment variable not set")
    return create_engine(dsn, echo=False)

engine = get_engine()
```

---

# 3. **LangChain v1 Tool: run_sql_query(sql: str)**

Uses `sqlmodel` session + SQLAlchemy `text()` for free-form SQL.

```python
from langchain.tools import tool
from sqlmodel import Session
from sqlalchemy import text
from textwrap import shorten

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
```

This cleanly uses:

* `sqlmodel.Session`
* SQLAlchemy `text()` for free SQL
* LangChain v1 `@tool` signature

Perfect for RAG-Agents & DB QA.

---

# 4. Use This Tool in a Gemini Agent (LangChain v1)

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)

agent = create_agent(
    model=llm,
    tools=[run_sql_query],
)
```

---

# 5. Invoke Agent

```python
result = agent.invoke({
    "messages": [
        HumanMessage(content="Show me the last 10 employees ordered by salary desc.")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
```

The LLM will autonomously call your SQLModel-powered `run_sql_query` tool.

---

# 6. Run
```
uv --project uv_env/ run python week_05/05_tools/main1_query_database.py 
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
```
