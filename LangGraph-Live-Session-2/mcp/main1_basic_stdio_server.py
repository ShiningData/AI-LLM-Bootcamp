"""
Basic stdio MCP server with math tools.

Shows how to:
- Create a simple MCP server using FastMCP
- Define basic math tools (add, multiply, subtract, power)
- Run the server with stdio transport (local subprocess)
- Use @mcp.tool() decorator for tool registration
"""
from mcp.server.fastmcp import FastMCP

# Create an MCP server named "Math"
# This name is exposed to MCP clients as the server identifier
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    result = a + b
    # Log to stderr instead of stdout to avoid JSON-RPC parsing issues
    import sys
    print(f"[MCP Math Server] Adding {a} + {b} = {result}", file=sys.stderr, flush=True)
    return result

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract second integer from the first and return the result."""
    result = a - b
    import sys
    print(f"[MCP Math Server] Subtracting {a} - {b} = {result}", file=sys.stderr, flush=True)
    return result

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    result = a * b
    import sys
    print(f"[MCP Math Server] Multiplying {a} * {b} = {result}", file=sys.stderr, flush=True)
    return result

@mcp.tool()
def power(base: int, exponent: int) -> int:
    """Calculate base raised to the power of exponent."""
    result = base ** exponent
    import sys
    print(f"[MCP Math Server] Power {base}^{exponent} = {result}", file=sys.stderr, flush=True)
    return result

if __name__ == "__main__":
    import sys
    print("[MCP Math Server] Starting stdio math server...", file=sys.stderr, flush=True)
    
    # Run the server using stdio transport
    # This means:
    # - The server reads JSON-RPC messages from stdin
    # - Sends responses back via stdout
    # - Perfect for local development and subprocess communication
    mcp.run(transport="stdio")