"""
Stateful MCP sessions with persistent context.

Shows how to:
- Create an MCP server that maintains state between tool calls
- Use client sessions for persistent server-side context
- Implement stateful operations like storing variables and calculations
- Demonstrate how session state survives across multiple tool invocations
"""
import asyncio
from pathlib import Path
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Global state dictionary for this demo server
# In production, you'd use a proper database or cache
server_state = {}

# Create an MCP server named "StatefulCalculator"
mcp = FastMCP("StatefulCalculator")

@dataclass
class CalculationState:
    """Track calculation state and variables."""
    stored_variables: dict[str, float]
    calculation_history: list[str]

@mcp.tool()
def store_number(name: str, value: float) -> str:
    """Store a number with a given name for later use."""
    server_state[name] = value
    return f"Stored {value} as '{name}'. Current storage: {list(server_state.keys())}"

@mcp.tool()
def get_number(name: str) -> float:
    """Retrieve a previously stored number by name."""
    if name not in server_state:
        raise ValueError(f"No number stored with name '{name}'. Available: {list(server_state.keys())}")
    return server_state[name]

@mcp.tool()
def calculate_with_stored(operation: str, stored_name: str, value: float) -> float:
    """
    Perform an operation between a stored number and a new value.
    
    Args:
        operation: 'add', 'subtract', 'multiply', or 'divide'
        stored_name: Name of the stored number to use
        value: The other value for the operation
    """
    if stored_name not in server_state:
        raise ValueError(f"No number stored with name '{stored_name}'")
    
    stored_value = server_state[stored_name]
    
    if operation == "add":
        result = stored_value + value
    elif operation == "subtract":
        result = stored_value - value
    elif operation == "multiply":
        result = stored_value * value
    elif operation == "divide":
        if value == 0:
            raise ValueError("Cannot divide by zero")
        result = stored_value / value
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result

@mcp.tool()
def list_stored() -> str:
    """List all currently stored numbers."""
    if not server_state:
        return "No numbers currently stored."
    
    items = [f"{name}: {value}" for name, value in server_state.items()]
    return f"Stored numbers: {', '.join(items)}"

@mcp.tool()
def clear_storage() -> str:
    """Clear all stored numbers."""
    count = len(server_state)
    server_state.clear()
    return f"Cleared {count} stored numbers."

# Function to run the stateful server (for testing)
async def start_stateful_server():
    """Start the stateful calculator server."""
    print("[Stateful Server] Starting stateful calculator server...")
    await mcp.run_stdio_async()

async def demonstrate_stateful_sessions():
    """Demonstrate stateful MCP sessions with LangChain."""
    print("=== Stateful MCP Sessions Example ===\n")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    server_script = script_dir / "main4_stateful_sessions.py"
    
    # Configure the stateful server
    client = MultiServerMCPClient({
        "calculator": {
            "transport": "stdio",
            "command": "python",
            "args": [str(server_script), "--server"]  # Special flag to run as server
        }
    })
    
    print("🔌 Connecting to stateful MCP server...")
    
    try:
        # Create a persistent session
        # This maintains server-side state throughout the session
        async with client.session("calculator") as session:
            print("📋 Created persistent session with stateful server")
            
            # Load tools within the session context
            tools = await load_mcp_tools(session)
            print(f"🛠️  Loaded {len(tools)} stateful tools:")
            for tool in tools:
                print(f"   - {tool.name}")
            print()
            
            # Get tool functions for direct calling
            tool_map = {tool.name: tool for tool in tools}
            
            # Example 1: Store some numbers
            print("📊 Step 1: Storing numbers...")
            result1 = await tool_map["store_number"].ainvoke({"name": "base_price", "value": 100.0})
            print(f"   {result1}")
            
            result2 = await tool_map["store_number"].ainvoke({"name": "tax_rate", "value": 0.15})
            print(f"   {result2}")
            
            result3 = await tool_map["store_number"].ainvoke({"name": "discount", "value": 20.0})
            print(f"   {result3}")
            print()
            
            # Example 2: List stored numbers
            print("📋 Step 2: Checking stored numbers...")
            stored_list = await tool_map["list_stored"].ainvoke({})
            print(f"   {stored_list}")
            print()
            
            # Example 3: Perform calculations with stored numbers
            print("🧮 Step 3: Calculating with stored numbers...")
            
            try:
                # Calculate tax: base_price * tax_rate
                tax_amount_result = await tool_map["calculate_with_stored"].ainvoke({
                    "operation": "multiply",
                    "stored_name": "base_price", 
                    "value": 0.15  # tax rate
                })
                tax_amount = float(tax_amount_result)
                print(f"   Tax amount: ${tax_amount:.2f}")
                
                # Store the tax amount
                await tool_map["store_number"].ainvoke({"name": "tax_amount", "value": tax_amount})
                
                # Calculate total: base_price + tax_amount
                subtotal_result = await tool_map["calculate_with_stored"].ainvoke({
                    "operation": "add",
                    "stored_name": "base_price",
                    "value": tax_amount
                })
                subtotal = float(subtotal_result)
                print(f"   Subtotal with tax: ${subtotal:.2f}")
                
                # Apply discount: subtotal - discount
                final_price_result = await tool_map["calculate_with_stored"].ainvoke({
                    "operation": "subtract",
                    "stored_name": "base_price",  # Using base price for stored value
                    "value": 20.0  # discount amount
                })
                final_price = float(final_price_result)
                print(f"   Final price after discount: ${final_price:.2f}")
                print()
                
            except Exception as calc_error:
                print(f"   ⚠️ Calculation error: {calc_error}")
                print("   Continuing with session persistence demo...")
                print()
            
            # Example 4: Show persistence within session
            print("💾 Step 4: Demonstrating session persistence...")
            final_stored = await tool_map["list_stored"].ainvoke({})
            print(f"   All stored values still available: {final_stored}")
            print()
            
        # After exiting the session context, the server state is cleaned up
        print("🔚 Session ended - server state cleaned up")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the stateful server script is executable")
        print("2. Install required packages: pip install langchain-mcp-adapters mcp")

async def main():
    """Main function to choose between server and client modes."""
    import sys
    
    if "--server" in sys.argv:
        # Run as server
        await start_stateful_server()
    else:
        # Run as client demo
        await demonstrate_stateful_sessions()

if __name__ == "__main__":
    asyncio.run(main())