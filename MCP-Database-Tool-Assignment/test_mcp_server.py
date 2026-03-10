"""
Proper MCP Database Server Test Script.
This uses the correct MCP protocol via langchain-mcp-adapters.
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_mcp_server():
    """Test the MCP database server using proper MCP protocol"""
    print("Testing MCP Database Server...")
    print("Make sure the server is running: python database_mcp_server.py")
    print("="*60)
    
    try:
        # Connect to MCP server using proper protocol
        client = MultiServerMCPClient({
            'database': {
                'transport': 'streamable_http',
                'url': 'http://localhost:8000/mcp'
            }
        })
        
        print("\n✅ Connected to MCP server successfully!")
        
        # Get available tools
        tools = await client.get_tools()
        print(f"\n✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description.split('.')[0]}...")
        
        # Test each tool
        print("\n" + "="*60)
        print("Testing individual tools:")
        
        # Test 1: Get database info
        print("\n1. Testing get_database_info...")
        db_info_tool = next(tool for tool in tools if tool.name == "get_database_info")
        result = await db_info_tool.ainvoke({})
        print(f"✅ SUCCESS: {result}")
        
        # Test 2: List tables
        print("\n2. Testing list_tables...")
        list_tables_tool = next(tool for tool in tools if tool.name == "list_tables")
        result = await list_tables_tool.ainvoke({})
        print(f"✅ SUCCESS: {result}")
        
        # Test 3: Get table schema
        print("\n3. Testing get_table_schema...")
        schema_tool = next(tool for tool in tools if tool.name == "get_table_schema")
        result = await schema_tool.ainvoke({"table_names": "artist, album"})
        print(f"✅ SUCCESS: Schema retrieved (length: {len(str(result))} chars)")
        
        # Test 4: Validate query
        print("\n4. Testing validate_query...")
        validate_tool = next(tool for tool in tools if tool.name == "validate_query")
        result = await validate_tool.ainvoke({"query": "SELECT * FROM artist LIMIT 5"})
        print(f"✅ SUCCESS: {result}")
        
        # Test 5: Execute query
        print("\n5. Testing execute_query...")
        execute_tool = next(tool for tool in tools if tool.name == "execute_query")
        result = await execute_tool.ainvoke({"query": "SELECT name FROM artist LIMIT 3"})
        print(f"✅ SUCCESS: {result}")
        
        print("\n" + "="*60)
        print("🎉 All tests passed! MCP Database Server is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the MCP server is running: python database_mcp_server.py")
        print("2. Check that langchain-mcp-adapters is installed: pip install langchain-mcp-adapters")
        print("3. Verify the database connection in the server")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())