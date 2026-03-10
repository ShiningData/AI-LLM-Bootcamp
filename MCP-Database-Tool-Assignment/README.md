# MCP Database Tool

This module converts the original database query tool to use Model Context Protocol (MCP) architecture, separating the database operations into a server and client.

## Architecture

- **MCP Server** (`database_mcp_server.py`): Provides database query capabilities as MCP tools
- **MCP Client** (`simple_mcp_client.py`): Uses the MCP server tools in a LangGraph workflow

## Files

1. `database_mcp_server.py` - MCP server that exposes database tools
2. `simple_mcp_client.py` - Client that mirrors the original main.py structure 
3. `database_mcp_client.py` - More advanced async client implementation
4. `requirements.txt` - Required dependencies

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file):
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. Start PostgreSQL with Docker:
   ```bash
   docker run --name chinook-postgres \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=Ankara06 \
     -e POSTGRES_DB=traindb \
     -p 5432:5432 \
     -d postgres:16
   ```

5. Wait a few seconds for PostgreSQL to initialize, then load the Chinook database:
   ```bash
   docker cp chinook_pg_serial_pk_proper_naming.sql chinook-postgres:/tmp/
   docker exec -i chinook-postgres psql -U postgres -d traindb -f /tmp/chinook_pg_serial_pk_proper_naming.sql
   ```

## Usage

### Step 1: Start the MCP Server
```bash
python database_mcp_server.py
```

This will start the MCP server on `http://localhost:8000/mcp`

### Step 2: Run the Database Query Client
In a separate terminal:
```bash
python database_query_client.py
```

## Available MCP Tools

The MCP server provides these tools:

- `list_tables()` - List all database tables
- `get_table_schema(table_names)` - Get schema for specified tables  
- `execute_query(query)` - Execute SELECT queries safely
- `validate_query(query)` - Validate SQL syntax and check for issues
- `get_database_info()` - Get database connection information

## Differences from Original

1. **Separation of Concerns**: Database operations are in a separate server
2. **Remote Access**: MCP server can be accessed remotely via HTTP
3. **Safety**: Built-in query validation and safety checks
4. **Scalability**: Multiple clients can connect to the same MCP server
5. **Modularity**: Database logic is reusable across different applications

## Example Queries

Once running, you can ask questions like:
- "Show me the top 5 artists"
- "How many albums are in the database?"
- "List customers from Canada"
- "Show tracks longer than 5 minutes"

The system will automatically:
1. List available tables
2. Get relevant table schemas
3. Generate appropriate SQL queries
4. Validate queries before execution
5. Execute and return results

## Grading Rubric (100 Points)

### 1. MCP Server (`database_mcp_server.py`) - 30 Points

| Criteria | Points | Description |
|----------|--------|-------------|
| FastMCP Setup | 5 | Proper MCP server initialization with `FastMCP("Database")` |
| `list_tables` tool | 5 | Returns all available table names |
| `get_table_schema` tool | 5 | Returns schema and sample rows for specified tables |
| `execute_query` tool | 5 | Executes SQL queries with SELECT-only restriction |
| `validate_query` tool | 5 | Validates queries for dangerous keywords and common mistakes |
| `get_database_info` tool | 5 | Returns database metadata (dialect, tables, connection status) |

### 2. MCP Client (`database_query_client.py`) - 30 Points

| Criteria | Points | Description |
|----------|--------|-------------|
| MCP Connection | 5 | `MultiServerMCPClient` with `streamable_http` transport |
| LangGraph Workflow | 10 | Complete workflow with `StateGraph`, nodes, and edges |
| LLM Integration | 5 | `init_chat_model` with tool binding |
| Conditional Edges | 5 | `should_continue` function for workflow control |
| Interactive Loop | 5 | User input handling with async streaming |

### 3. Safety & Error Handling - 15 Points

| Criteria | Points | Description |
|----------|--------|-------------|
| SELECT-only Restriction | 5 | Blocks INSERT, UPDATE, DELETE, DROP queries |
| Dangerous Keyword Detection | 5 | Validates against DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE |
| Try-Catch Error Handling | 5 | Graceful error messages in all tools |

### 4. Test Suite (`test_mcp_server.py`) - 10 Points

| Criteria | Points | Description |
|----------|--------|-------------|
| MCP Connection Test | 2 | Verifies server connectivity |
| All 5 Tools Tested | 5 | Individual test for each MCP tool |
| Error Handling & Output | 3 | Clear success/failure messages with troubleshooting |

### 5. Documentation & Configuration - 15 Points

| Criteria | Points | Description |
|----------|--------|-------------|
| `README.md` | 5 | Setup instructions, usage, and tool descriptions |
| `requirements.txt` | 5 | All dependencies with correct versions |
| Docker Setup | 5 | PostgreSQL container with Chinook database |

### Total: 100 Points