## Streaming
Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

- What’s possible with LangChain streaming:
    - Stream agent progress — get state updates after each agent step.
    - Stream LLM tokens — stream language model tokens as they’re generated.
    - Stream custom updates — emit user-defined signals (e.g., "Fetched 10/100 records").
    - Stream multiple modes — choose from updates (agent progress), messages (LLM tokens + metadata), or custom (arbitrary user data).

### Stream Agent progress
- To stream agent progress, use the stream or astream methods with stream_mode="updates". This emits an event after every agent step.
- For example, if you have an agent that calls a tool once, you should see the following updates:
    - LLM node: AIMessage with tool call requests
    - Tool node: ToolMessage with execution result
    - LLM node: Final AI response

```python
from langchain.agents import create_agent
# create_agent: builds a LangChain v1 agent with tools and streaming support.


# Simple tool the agent can call
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Create an agent that can call the get_weather tool
agent = create_agent(
    model="gpt-5-nano",   # LLM used for reasoning + deciding tool calls
    tools=[get_weather],
)

# Stream *agent progress* (node-level updates) instead of final result only
for chunk in agent.stream(  # stream() yields chunks after each step
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",  # "updates" → per-node state changes (model, tools, etc.)
):
    # Each chunk is a dict: { node_name -> node_data }
    for step, data in chunk.items():
        print(f"step: {step}")  # e.g. "model", "tools"
        # The last message in that node’s messages list
        print(f"content: {data['messages'][-1].content_blocks}")
```

### Stream LLM tokens
- To stream tokens as they are produced by the LLM, use stream_mode="messages". Below you can see the output of the agent streaming tool calls and the final response.

```python
from langchain.agents import create_agent
# create_agent: same as above, but we’ll stream *tokens* this time.


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

# Stream *LLM messages/tokens* as they are generated
for token, metadata in agent.stream(  # stream() now yields (token, metadata)
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",  # "messages" → token / content chunk streaming
):
    # Which node produced this token? (e.g. "model", "tools")
    print(f"node: {metadata['langgraph_node']}")
    # token.content_blocks contains the incremental data (tool_call chunks, text chunks, etc.)
    print(f"content: {token.content_blocks}")
    print("\n")
```

### Custom updates
- To stream updates from tools as they are executed, you can use get_stream_writer.
```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer  # Utility to emit custom streamed data
# get_stream_writer: returns a writer function bound to the current streaming context.


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()  # Get a writer for "custom" stream data

    # Emit arbitrary progress messages while the tool runs
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    # Normal tool return value still goes back to the agent
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # Model used in the docs example
    tools=[get_weather],
)

# Stream ONLY the "custom" channel emitted by get_stream_writer()
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom",  # Focus on custom messages emitted by tools
):
    print(chunk)  # Each chunk is whatever the writer() was called with (strings here)
```

### Stream multiple modes
- You can specify multiple streaming modes by passing stream mode as a list: stream_mode=["updates", "custom"]:

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
# Same idea as before but now we stream both *updates* and *custom* channels.


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # Emit arbitrary, user-defined progress messages
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

# Here stream_mode is a LIST → we get multiple channels interleaved
for stream_mode, chunk in agent.stream(  # stream() yields (stream_mode, chunk)
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],  # Receive both node updates and custom logs
):
    print(f"stream_mode: {stream_mode}")  # "updates" or "custom"
    print(f"content: {chunk}")            # Content depends on the mode
    print("\n")
```

### Disable streaming
- In some applications you might need to disable streaming of individual tokens for a given model.
- This is useful in multi-agent systems to control which agents stream their output.

- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/07_streaming/main1_streaming_example.py
```
- Output
```
🚀 DEMO 1: Streaming Agent Progress (updates mode)
============================================================
📍 Step: model
   Content: 

📍 Step: tools
   Content: The weather in Paris is sunny and 22°C!

📍 Step: model
   Content: The weather in Paris is sunny and 22°C!

🚀 DEMO 1: Streaming Agent Progress (updates mode)
============================================================
📍 Step: model
   Content: 

📍 Step: tools
   Content: The weather in Paris is sunny and 22°C!

📍 Step: model
   Content: The weather in Paris is sunny and 22°C!
```
#### DEMO 1 (Agent Progress): Shows each step the agent takes:
  - model step: Agent thinks and decides to call tools
  - tools step: Tools execute and return results
  - model step: Agent processes tool results and gives final answer

#### DEMO 2 (Token Streaming): Shows tokens as the LLM generates them:
  - Each token/word appears as it's being generated
  - Like watching someone type in real-time

#### DEMO 3 (Custom Updates): Shows progress messages from inside tools:
  - ⏰ Checking time for timezone... - tool is starting
  - 🌍 Converting to local time... - tool is working
  - These come from the writer() calls in your tools

#### DEMO 4 (Multiple Modes): Combines agent steps + custom updates:
  - You see both what the agent is doing AND what the tools are reporting
  - Interleaved real-time updates from different sources

  The benefit is user experience - instead of waiting 5-10 seconds for a final answer, users see:
  - "Agent is thinking..."
  - "Looking up weather data..."
  - "Got weather data, now checking time..."
  - "Final answer: ..."

  This makes apps feel more responsive and gives users confidence that something is happening, not just hanging.