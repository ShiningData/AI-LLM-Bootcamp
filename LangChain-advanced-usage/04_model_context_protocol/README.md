# Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools and context to LLMs. LangChain agents can use tools defined on MCP servers using the langchain-mcp-adapters library.

These examples show how to:
- Create MCP servers with different transport types
- Connect LangChain agents to MCP tools
- Handle stateful vs stateless tool interactions
- Implement streaming and real-time capabilities

## 🛠️ Prerequisites

```bash
# Install required packages
pip install mcp langchain-mcp-adapters python-dotenv

# Set Google API key (for examples that use LangChain agents)
export GOOGLE_API_KEY="your-api-key-here"
```

## 📁 Examples

### 1️⃣ [main1_basic_stdio_server.py](./main1_basic_stdio_server.py)
**Basic stdio MCP Server with Math Tools**

**What it is**: A simple MCP server that provides math operations (add, subtract, multiply, power) using stdio transport.

**How to run**:
```bash
# Run as standalone server (will listen for MCP requests on stdin)
uv --project uv_env/ run python main1_basic_stdio_server.py
```

**What to expect**: 
```
[MCP Math Server] Starting stdio math server...
# Server waits for JSON-RPC messages on stdin
```

**How to interact**: This server is designed to be used by MCP clients (like main3), not directly. When used with a client, you'll see:
```
[MCP Math Server] Adding 25 + 15 = 40
[MCP Math Server] Multiplying 40 * 3 = 120
```

---

### 2️⃣ [main2_http_server.py](./main2_http_server.py)
**HTTP MCP Server with Weather Tools**

**What it is**: An HTTP-based MCP server providing weather tools (get_weather, get_forecast, get_weather_alerts).

**How to run**:
```bash
# Start HTTP server (runs in foreground)
uv --project uv_env/ run python main2_http_server.py
```

**What to expect**:
```
[MCP Weather Server] Starting HTTP weather server on localhost:8000...
[MCP Weather Server] MCP endpoint will be available at http://localhost:8000/mcp
INFO:     Started server process [1234]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**How to interact**: 
- **Keep this server running** in one terminal
- Use other examples (like main3_multi_server_client.py) to connect to it
- When tools are called, you'll see: `[MCP Weather Server] Weather in Tokyo: Sunny, 22°C`

**Note**: 404 errors for `/docs` and `/favicon.ico` are normal - the MCP endpoint is at `/mcp`.

---

### 3️⃣ [main3_multi_server_client.py](./main3_multi_server_client.py) ⭐ **START HERE**
**Multi-Server MCP Client with LangChain Agent**

**What it is**: A LangChain agent that connects to both the math server (stdio) and weather server (HTTP) simultaneously.

**Prerequisites**:
```bash
# 1. Start the weather server in another terminal:
uv --project uv_env/ run python main2_http_server.py

# 2. Set Google API key:
export GOOGLE_API_KEY="your-api-key-here"
```

**How to run**:
```bash
# Run the multi-server client (in a new terminal)
uv --project uv_env/ run python main3_multi_server_client.py
```

**What to expect**:
```
=== Multi-Server MCP Client Example ===

🔌 Connecting to MCP servers...
   - Math server (stdio): /path/to/main1_basic_stdio_server.py
   - Weather server (HTTP): http://localhost:8000/mcp

🛠️  Loading tools from MCP servers...
   Found 7 tools:
   - add: Add two integers and return the result.
   - subtract: Subtract second integer from the first and return the result.
   - multiply: Multiply two integers and return the result.
   - power: Calculate base raised to the power of exponent.
   - get_weather: Get current weather for a given location.
   - get_forecast: Get weather forecast for a given location and number of days.
   - get_weather_alerts: Get weather alerts and warnings for a given location.

🤖 Agent created successfully with MCP tools!
==================================================

📊 Example 1: Math calculation
> Entering new AgentExecutor chain...
[Agent performs calculation using math tools]
Result: The calculation (25 + 15) × 3 equals 120.

🌤️  Example 2: Weather query
> Entering new AgentExecutor chain...
[Agent calls weather tools]
Result: Weather in Tokyo: Sunny, 22°C
```

**How to interact**: This example runs automatically with preset queries, but you can modify the queries in the code.

---

### 4️⃣ [main4_stateful_sessions.py](./main4_stateful_sessions.py)
**Stateful MCP Sessions**

**What it is**: Demonstrates how MCP servers can maintain state between tool calls (like storing variables).

**How to run**:
```bash
# Run the stateful session demo
python main4_stateful_sessions.py
```

**What to expect**:
```
=== Stateful MCP Sessions Example ===

🔌 Connecting to stateful MCP server...
📋 Created persistent session with stateful server
🛠️  Loaded 5 stateful tools:
   - store_number
   - get_number
   - calculate_with_stored
   - list_stored
   - clear_storage

📊 Step 1: Storing numbers...
   Stored 100.0 as 'base_price'. Current storage: ['base_price']
   Stored 0.15 as 'tax_rate'. Current storage: ['base_price', 'tax_rate']
   Stored 20.0 as 'discount'. Current storage: ['base_price', 'tax_rate', 'discount']

📋 Step 2: Checking stored numbers...
   Stored numbers: base_price: 100.0, tax_rate: 0.15, discount: 20.0

🧮 Step 3: Calculating with stored numbers...
   Tax amount: $15.00
   Subtotal with tax: $115.00
   Final price after discount: $80.00

💾 Step 4: Demonstrating session persistence...
   All stored values still available: Stored numbers: base_price: 100.0, tax_rate: 0.15, discount: 20.0, tax_amount: 15.0

🔚 Session ended - server state cleaned up
```

**Note**: You may see INFO logs about "Processing request of type CallToolRequest" - these are normal and indicate the MCP server is handling requests correctly.

**How to interact**: This example runs automatically with preset operations, demonstrating how state persists within a session.

---

### 5️⃣ [main5_sse_streaming.py](./main5_sse_streaming.py)
**SSE Transport with Streaming**

**What it is**: An MCP server using Server-Sent Events for real-time streaming of long-running operations.

**How to run**:
```bash
# Start SSE streaming server
uv --project uv_env/ run python main5_sse_streaming.py
```

**What to expect**:
```
[Streaming Server] Starting SSE-enabled streaming server...
[Streaming Server] This server provides real-time streaming capabilities
[Streaming Server] Perfect for long-running operations and progress tracking

# Server starts and waits for SSE connections
```

**How to interact**: This server is designed for SSE clients. When tools are called, you'll see progress updates:
```
[Streaming Server] Starting to process 1000 records...
[Streaming Server] Progress: 10.0% (100/1000)
[Streaming Server] Progress: 20.0% (200/1000)
[Streaming Server] Progress: 30.0% (300/1000)
...
[Streaming Server] Finished processing 1000 records
```

**Note**: Building SSE clients requires special handling - this example shows the server side.

## 🚀 Quick Start Guide

**Want to see MCP in action? Follow these steps:**

1. **First, start the HTTP weather server** (keep it running):
   ```bash
   # Terminal 1 - Weather Server
   uv --project uv_env/ run python main2_http_server.py
   ```

2. **Set your Google API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. **Run the multi-server demo** (this is the most exciting example):
   ```bash
   # Terminal 2 - Client Demo
   uv --project uv_env/ run python main3_multi_server_client.py
   ```

4. **Watch the magic happen**: The LangChain agent will automatically:
   - Connect to both math (stdio) and weather (HTTP) servers
   - Discover all available tools
   - Solve math problems using the math server
   - Get weather information using the weather server
   - Show you how multiple MCP servers work together

**That's it!** You've seen the full MCP workflow in action.

**Next steps:**
- Try the stateful sessions: `python main4_stateful_sessions.py`
- Explore streaming: `python main5_sse_streaming.py`
- Modify the examples to create your own tools

## 🚚 Transport Types

| Transport | Example | Best For | Pros | Cons |
|-----------|---------|----------|------|------|
| **stdio** | main1, main4 | Local tools, simple use | Very fast, no network | Only local, not scalable |
| **HTTP** | main2, main3 | Remote servers, production | Scalable, cloud-ready | More latency |
| **SSE** | main5 | Real-time updates, streaming | Streaming, great UX | One-way stream, complex setup |

## 📖 Key Learning Points

### 1. **Server Creation**
- Use `FastMCP("ServerName")` to create servers
- Register tools with `@mcp.tool()` decorator
- Choose appropriate transport for your use case

### 2. **Client Integration**
- `MultiServerMCPClient` connects to multiple servers
- `client.get_tools()` discovers and loads all available tools
- LangChain agents can use MCP tools seamlessly

### 3. **Stateful vs Stateless**
- **Stateless**: Each tool call is independent (main1, main2, main5)
- **Stateful**: Server maintains context between calls (main4)
- Use `client.session()` for stateful interactions

### 4. **Production Considerations**
- stdio: Development, local tools
- HTTP: Production, remote access, load balancing
- SSE: Real-time features, progress tracking

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install mcp langchain-mcp-adapters python-dotenv
   ```

2. **Connection Refused (HTTP server)**
   - Make sure the server is running: `python main2_http_server.py`
   - Check port 8000 is available

3. **Google API Key Missing**
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

4. **Async/Await Issues**
   - MCP operations are async - use `await` and `asyncio.run()`

### Debug Tips

- Check server logs for error messages
- Use `print()` statements in tools for debugging
- Verify transport configurations match between client/server
- Test tools individually before integrating with agents

## 📚 Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/docs/getting-started/intro)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/mcp)
- [Transport Types Guide](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)

## 🎯 Next Steps

After working through these examples:

1. **Create your own MCP server** with domain-specific tools
2. **Integrate with production LLM applications**
3. **Experiment with different transport types**
4. **Build stateful workflows** for complex use cases
5. **Add authentication and security** for production servers
