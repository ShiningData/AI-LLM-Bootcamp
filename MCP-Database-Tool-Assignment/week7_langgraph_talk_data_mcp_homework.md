# Week 7 Assignment: Converting Database Query Tool to MCP Architecture

## Assignment Overview

**Objective**: Convert the existing database query tool from a monolithic LangChain application to a distributed Model Context Protocol (MCP) architecture.

**Original File**: `week_07_langgraph/database_query_tool/main.py`  
**Reference Example**: `mcp/main2_http_server.py`

## Assignment Requirements

1. **Create MCP Server**: Convert database operations to MCP tools
2. **Create MCP Client**: Maintain original workflow using MCP tools
3. **Preserve Functionality**: Same user experience as original
4. **Add Safety**: Implement query validation and security measures
5. **Enable Remote Access**: Use HTTP transport for distributed access
6. **Database on Docker**: Use same docker run and sql script to create chinook database

## Solution Implementation

### Architecture Design

**Before (Monolithic)**:
```
User Input → LangChain SQL Toolkit → Database → Results
```

**After (MCP Distributed)**:
```
User Input → MCP Client → HTTP → MCP Server → Database → Results
```

### Created Files

1. **`database_mcp_server.py`** - MCP server exposing database tools
2. **`database_query_client.py`** - Interactive client maintaining original workflow
3. **`test_mcp_server.py`** - Comprehensive test suite for MCP tools
4. **`requirements.txt`** - All necessary dependencies
5. **`README.md`** - Complete usage documentation

### MCP Tools Implemented

| Tool Name | Purpose | Safety Features |
|-----------|---------|----------------|
| `list_tables` | Get all database tables | Read-only operation |
| `get_table_schema` | Retrieve table schemas | Limited to specified tables |
| `execute_query` | Run SQL queries | SELECT-only restriction |
| `validate_query` | Check query syntax | Pre-execution validation |
| `get_database_info` | Database metadata | Connection status info |

### Security Enhancements

- **Query Restrictions**: Only SELECT statements allowed
- **Input Validation**: SQL injection prevention
- **Error Handling**: Graceful failure with informative messages
- **Connection Safety**: Proper async resource management

### MCP Protocol Integration

**Transport**: `streamable-http` for remote connectivity  
**Protocol**: JSON-RPC over HTTP with Server-Sent Events  
**Library**: `langchain-mcp-adapters` for proper MCP integration  
**Port**: 8000 (configurable via environment)

## Key Technical Achievements

### 1. **Seamless Migration**
- Preserved exact LangGraph workflow structure
- Maintained identical user interaction patterns
- Same system prompts and decision logic

### 2. **Enhanced Modularity**
```python
# Original: Direct database access
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# MCP: Remote tool access
client = MultiServerMCPClient({
    "database": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp"
    }
})
tools = await client.get_tools()
```

### 3. **Improved Safety**
```python
@mcp.tool()
async def execute_query(query: str) -> str:
    # Safety check - only allow SELECT statements
    query_upper = query.strip().upper()
    if not query_upper.startswith('SELECT'):
        return "Error: Only SELECT queries allowed for safety"
```

### 4. **Better Error Handling**
- Async operations with proper exception handling
- Detailed error messages for debugging
- Graceful degradation on connection failures

## Testing & Validation

### Comprehensive Test Suite
```bash
python test_mcp_server.py
```

**Test Results**:
- MCP server connection
- All 5 tools discovered and functional
- Database queries executing correctly
- Query validation working
- Error handling verified

### Interactive Testing
```bash
# Terminal 1: Start server
python database_mcp_server.py

# Terminal 2: Run client
python database_query_client.py
```