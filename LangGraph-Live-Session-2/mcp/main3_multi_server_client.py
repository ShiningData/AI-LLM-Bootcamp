"""
Multi-server MCP client with LangChain integration.

Shows how to:
- Connect to multiple MCP servers simultaneously
- Use different transport types (stdio and HTTP) in one client
- Create a LangChain agent that can use tools from multiple servers
- Automatic tool discovery and integration
"""
import asyncio
import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

async def demonstrate_multi_server_mcp():
    """Demonstrate connecting to multiple MCP servers with different transports."""
    print("=== Multi-Server MCP Client Example ===\n")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    print(f"script_dir: {script_dir}")
    math_server_path = script_dir / "main1_basic_stdio_server.py"
    
    # Configure multiple MCP servers with different transports
    client = MultiServerMCPClient({
        # Local math server using stdio transport
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [str(math_server_path)]
        },
        
        # Remote weather server using HTTP transport
        # Note: You need to run main2_http_server.py separately
        "weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp"
        }
    })
    
    print(" Connecting to MCP servers...")
    print(f"   - Math server (stdio): {math_server_path}")
    print("   - Weather server (HTTP): http://localhost:8000/mcp")
    print()
    
    try:
        # Load tools from all configured MCP servers
        print("  Loading tools from MCP servers...")
        tools = await client.get_tools()
        
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        print()
        
        # Create the agent with MCP tools (using same pattern as context engineering examples)
        agent = create_agent(model, tools)
        
        print(" Agent created successfully with MCP tools!")
        print("=" * 50)
        
        # Example 1: Using math tools
        print("\n Example 1: Math calculation")
        math_response = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": "What is (25 + 15) multiplied by 3?"}
            ]
        })
        print(f"Result: {math_response}")
        
        print("\n" + "=" * 50)
        
        # Example 2: Using weather tools  
        print("\n  Example 2: Weather query")
        weather_response = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": "What's the weather like in Tokyo?"}
            ]
        })
        print(f"Result: {weather_response}")
        
        print("\n" + "=" * 50)
        
        # Example 3: Complex query using both types of tools
        print("\n Example 3: Combined calculation and weather")
        combined_response = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": "If the temperature in London is 20°C and in Paris it's 5°C warmer, what's the total temperature if I add them together and multiply by 2?"}
            ]
        })
        print(f"Result: {combined_response}")
        
    except Exception as e:
        print(f" Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have GOOGLE_API_KEY set in your environment")
        print("2. Start the weather server: python main2_http_server.py")
        print("3. Install required packages: pip install langchain-mcp-adapters python-dotenv")
        
    finally:
        # Clean up connections
        print("\n Closing MCP client connections...")
        # The client will automatically clean up when it goes out of scope

if __name__ == "__main__":
    asyncio.run(demonstrate_multi_server_mcp())