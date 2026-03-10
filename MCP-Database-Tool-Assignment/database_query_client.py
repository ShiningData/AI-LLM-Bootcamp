"""
Correct MCP Database Client using langchain-mcp-adapters.

This version properly connects to the MCP database server using the official
langchain-mcp-adapters library and follows the same pattern as the multi-server example.
"""
import asyncio
import os
from typing import Literal
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize LLM
model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Global MCP client
mcp_client = None
tools = []

async def initialize_mcp_client():
    """Initialize connection to MCP database server"""
    global mcp_client, tools
    
    try:
        # Configure MCP client to connect to database server
        mcp_client = MultiServerMCPClient({
            "database": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp"
            }
        })
        
        print("[MCP Client] Connecting to database server...")
        
        # Load tools from MCP server
        tools = await mcp_client.get_tools()
        
        print(f"[MCP Client] Connected! Found {len(tools)} database tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return True
    except Exception as e:
        print(f"[MCP Client] Failed to connect: {e}")
        return False

# Define workflow nodes similar to original
def list_tables(state: MessagesState):
    """Automatically lists all available database tables"""
    tool_call = {
        "name": "list_tables",
        "args": {},
        "id": "list_tables_001",
        "type": "tool_call",
    }
    
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    return {"messages": [tool_call_message]}

def call_get_schema(state: MessagesState):
    """Get schema for relevant tables based on user query"""
    # Extract user query to determine relevant tables
    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content.lower()
            break
    
    # Simple heuristic to determine relevant tables
    table_keywords = {
        'artist': 'artist',
        'album': 'album',
        'track': 'track', 
        'customer': 'customer',
        'invoice': 'invoice',
        'employee': 'employee',
        'film': 'film',
        'actor': 'actor'
    }
    
    relevant_tables = []
    for keyword, table in table_keywords.items():
        if keyword in user_query:
            relevant_tables.append(table)
    
    # Default to common tables if none found
    if not relevant_tables:
        relevant_tables = ['artist', 'album', 'track']
    
    # Create tool call to get schema
    tool_call = {
        "name": "get_table_schema",
        "args": {"table_names": ", ".join(relevant_tables)},
        "id": "get_schema_001",
        "type": "tool_call",
    }
    
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    return {"messages": [tool_call_message]}

# Generate query using LLM
generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct PostgreSQL query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""

def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    
    # Find the execute_query tool
    execute_tool = next((tool for tool in tools if tool.name == "execute_query"), None)
    if execute_tool:
        llm_with_tools = model.bind_tools([execute_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}
    else:
        return {"messages": [AIMessage("Error: execute_query tool not found")]}

# Query validation
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the PostgreSQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""

def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }
    
    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    
    # Find the execute_query tool
    execute_tool = next((tool for tool in tools if tool.name == "execute_query"), None)
    if execute_tool:
        llm_with_tools = model.bind_tools([execute_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}
    else:
        return {"messages": [AIMessage("Error: execute_query tool not found")]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"

async def build_workflow():
    """Build the LangGraph workflow"""
    # Create tool nodes for the MCP tools
    list_tables_tool = next((tool for tool in tools if tool.name == "list_tables"), None)
    get_schema_tool = next((tool for tool in tools if tool.name == "get_table_schema"), None)
    execute_tool = next((tool for tool in tools if tool.name == "execute_query"), None)
    
    if not all([list_tables_tool, get_schema_tool, execute_tool]):
        raise ValueError("Required MCP tools not found")
    
    list_tables_node = ToolNode([list_tables_tool], name="list_tables_execute")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema") 
    run_query_node = ToolNode([execute_tool], name="run_query")
    
    # Build the graph - same structure as original
    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node("list_tables_execute", list_tables_node)
    builder.add_node(call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node("run_query", run_query_node)
    
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "list_tables_execute")
    builder.add_edge("list_tables_execute", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges(
        "generate_query",
        should_continue,
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    
    return builder.compile()

async def main():
    """Main application loop"""
    print("[MCP Database Client] Starting MCP database client...")
    print("[MCP Database Client] Make sure the MCP database server is running on localhost:8000")
    
    # Initialize MCP connection
    if not await initialize_mcp_client():
        print("[MCP Database Client] Failed to connect to MCP server.")
        print("[MCP Database Client] Please start the server: python database_mcp_server.py")
        return
    
    try:
        # Build workflow
        agent = await build_workflow()
        
        print("\n[MCP Database Client] Ready! Ask database questions or type 'exit' to quit.")
        
        # Main interaction loop
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break
            
            print(f"\nProcessing: {user_input}\n")
            
            try:
                async for step in agent.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    stream_mode="values",
                ):
                    if step["messages"]:
                        last_message = step["messages"][-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            print(f"Step: {last_message.content}")
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "="*50)
            
    except KeyboardInterrupt:
        print("\n[MCP Database Client] Shutting down...")
    finally:
        # Cleanup will happen automatically when mcp_client goes out of scope
        print("[MCP Database Client] Closed.")

if __name__ == "__main__":
    # First run: python database_mcp_server.py
    # Then run: python correct_mcp_client.py
    asyncio.run(main())