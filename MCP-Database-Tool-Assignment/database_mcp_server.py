"""
MCP Database Server - Provides SQL database query capabilities as MCP tools.

Shows how to:
- Convert LangChain SQL toolkit into MCP tools
- Create async database operations
- Provide database schema inspection and query execution
- Handle database errors properly
- Use streamable-http transport for remote connectivity
"""
import asyncio
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Create an MCP server named "Database"
mcp = FastMCP("Database")

# Initialize database connection
db = SQLDatabase.from_uri("postgresql://postgres:Ankara06@localhost:5432/traindb")

@mcp.tool()
async def list_tables() -> str:
    """
    List all available tables in the database.
    Returns a comma-separated list of table names.
    """
    await asyncio.sleep(0.1)  # Simulate async operation
    try:
        tables = db.get_usable_table_names()
        result = ", ".join(tables)
        print(f"[MCP Database Server] Listed {len(tables)} tables")
        return result
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        print(f"[MCP Database Server] {error_msg}")
        return error_msg

@mcp.tool()
async def get_table_schema(table_names: str) -> str:
    """
    Get the schema and sample rows for specified tables.
    
    Args:
        table_names: Comma-separated list of table names to inspect
    """
    await asyncio.sleep(0.2)  # Simulate async operation
    try:
        # Clean up table names (remove spaces, split by comma)
        tables = [name.strip() for name in table_names.split(",") if name.strip()]
        
        if not tables:
            return "Error: No table names provided"
            
        result = db.get_table_info_no_throw(tables)
        print(f"[MCP Database Server] Retrieved schema for tables: {', '.join(tables)}")
        return result
    except Exception as e:
        error_msg = f"Error getting table schema: {str(e)}"
        print(f"[MCP Database Server] {error_msg}")
        return error_msg

@mcp.tool()
async def execute_query(query: str) -> str:
    """
    Execute a SQL query on the database.
    
    Args:
        query: The SQL query to execute (SELECT statements only for safety)
    
    Returns:
        Query results or error message
    """
    await asyncio.sleep(0.3)  # Simulate async operation
    try:
        # Basic safety check - only allow SELECT statements
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for safety reasons"
            
        # Execute the query
        result = db.run(query)
        print(f"[MCP Database Server] Executed query: {query[:50]}...")
        return str(result)
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(f"[MCP Database Server] {error_msg}")
        return error_msg

@mcp.tool()
async def validate_query(query: str) -> str:
    """
    Validate a SQL query without executing it.
    Checks for common SQL mistakes and syntax issues.
    
    Args:
        query: The SQL query to validate
    """
    await asyncio.sleep(0.1)
    try:
        # Basic validation checks
        query_stripped = query.strip()
        if not query_stripped:
            return "Error: Empty query provided"
            
        query_upper = query_stripped.upper()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return f"Warning: Query contains potentially dangerous keyword: {keyword}"
        
        # Check for common mistakes
        issues = []
        
        if 'NOT IN' in query_upper and 'NULL' in query_upper:
            issues.append("Warning: NOT IN with NULL values may not work as expected")
            
        if 'UNION' in query_upper and 'UNION ALL' not in query_upper:
            issues.append("Consider using UNION ALL if duplicates are acceptable (better performance)")
            
        if not query_stripped.endswith(';'):
            issues.append("Consider adding semicolon at the end of the query")
            
        if issues:
            return "Query validation completed. Issues found:\n" + "\n".join(issues)
        else:
            return "Query validation completed. No issues found."
            
    except Exception as e:
        return f"Error validating query: {str(e)}"

@mcp.tool()
async def get_database_info() -> str:
    """
    Get general information about the database including dialect and connection status.
    """
    await asyncio.sleep(0.1)
    try:
        info = f"""Database Information:
- Dialect: {db.dialect}
- Total Tables: {len(db.get_usable_table_names())}
- Connection Status: Active
- Database URI: {db._engine.url.database}@{db._engine.url.host}"""
        
        print("[MCP Database Server] Retrieved database info")
        return info
    except Exception as e:
        error_msg = f"Error getting database info: {str(e)}"
        print(f"[MCP Database Server] {error_msg}")
        return error_msg

if __name__ == "__main__":
    print("[MCP Database Server] Starting HTTP database server on localhost:8000...")
    print("[MCP Database Server] MCP endpoint will be available at http://localhost:8000/mcp")
    print(f"[MCP Database Server] Connected to database: {db.dialect}")
    print(f"[MCP Database Server] Available tables: {db.get_usable_table_names()}")
    
    # Run the server using streamable-http transport
    # FastMCP runs on port 8000 by default
    # To use a different port, set MCP_SERVER_PORT environment variable
    mcp.run(transport="streamable-http")