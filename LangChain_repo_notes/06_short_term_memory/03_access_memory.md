## Access memory
You can access and modify the short-term memory (state) of an agent in several ways:
​
### 1. Tools
#### Read short-term memory in a tool
Access short term memory (state) in a tool using the ToolRuntime parameter.
The tool_runtime parameter is hidden from the tool signature (so the model doesn’t see it), but the tool can access the state through it.
```python
from langchain.agents import create_agent, AgentState
# create_agent: builds a LangChain v1 agent.
# AgentState: the base class for agent-level state (messages + custom fields).

from langchain.tools import tool, ToolRuntime
# @tool decorator: defines a tool the agent can call.
# ToolRuntime: provides access to the agent's current state inside the tool.


# -----------------------------------------------------------
# CUSTOM STATE CLASS
# Extends AgentState to include additional fields (user_id).
# This becomes part of the agent's memory / runtime state.
# -----------------------------------------------------------
class CustomState(AgentState):
    user_id: str   # We add our own field to the agent state.


# -----------------------------------------------------------
# TOOL DEFINITION
# This tool reads the custom state and returns user info.
# -----------------------------------------------------------
@tool
def get_user_info(
    runtime: ToolRuntime   # ToolRuntime gives access to runtime.state
) -> str:
    """Look up user info."""
    
    # Access our custom state field "user_id"
    user_id = runtime.state["user_id"]

    # Return a matching result
    return "User is John Smith" if user_id == "user_123" else "Unknown user"


# -----------------------------------------------------------
# CREATE AGENT WITH CUSTOM STATE SCHEMA
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",     # The LLM used by the agent
    tools=[get_user_info],  # The agent is allowed to call get_user_info()
    state_schema=CustomState,  # IMPORTANT: enables custom agent state
)


# -----------------------------------------------------------
# INVOKE THE AGENT
# Supply:
# - A user message
# - A custom state field: user_id="user_123"
# The tool will read this custom “user_id” from runtime.state.
# -----------------------------------------------------------
result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"   # Passed into CustomState.user_id
})

# -----------------------------------------------------------
# PRINT FINAL LLM RESPONSE
# The model should return:
# "User is John Smith."
# -----------------------------------------------------------
print(result["messages"][-1].content)
# > User is John Smith.
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/06_short_term_memory/main5_tool_reading_memory.py
```

#### Write short-term memory from tools
- To modify the agent’s short-term memory (state) during execution, you can return state updates directly from the tools.
This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.
```python
from langchain.tools import tool, ToolRuntime
# @tool: decorator for defining agent tools
# ToolRuntime: gives tools access to both context and state at runtime.

from langchain_core.runnables import RunnableConfig
# RunnableConfig: configuration object for passing thread_id, metadata, etc.

from langchain.messages import ToolMessage
# ToolMessage: message produced by tools, inserted into conversation history.

from langchain.agents import create_agent, AgentState
# create_agent: builds a LangChain v1 agent
# AgentState: base class for agent memory (messages + custom fields)

from langgraph.types import Command
# Command: allows a tool to tell the agent to UPDATE STATE or MESSAGES.

from pydantic import BaseModel
# BaseModel: used for defining typed context schemas.


# -----------------------------------------------------------
# CUSTOM AGENT STATE
# This is stored in agent memory and persists between calls.
# -----------------------------------------------------------
class CustomState(AgentState):  
    user_name: str   # This will store the user's resolved name.


# -----------------------------------------------------------
# CUSTOM CONTEXT (NON-PERSISTENT INPUT)
# Context is passed per-request and NOT saved like state.
# Useful for transient metadata (user_id, tenant_id, language, etc.)
# -----------------------------------------------------------
class CustomContext(BaseModel):
    user_id: str   # Provided at invocation time, read by tools.


# -----------------------------------------------------------
# TOOL 1: update_user_info
# - Reads user_id from runtime.context
# - Determines the name
# - Returns a Command to UPDATE the agent's STATE + message history
# -----------------------------------------------------------
@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    
    # Read user_id from request context
    user_id = runtime.context.user_id 
    
    # Resolve name based on user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    
    # Command allows tools to update:
    # - The custom state (user_name)
    # - The message history (add a ToolMessage)
    return Command(update={  
        "user_name": name,   # Update CustomState.user_name
        
        "messages": [  # Add a success message to the conversation history
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


# -----------------------------------------------------------
# TOOL 2: greet
# - Reads user_name from AGENT STATE (not context)
# - Generates a greeting
# -----------------------------------------------------------
@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """Use this to greet the user once you found their info."""
    
    # Retrieve the name from agent state
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"


# -----------------------------------------------------------
# CREATE THE AGENT
# - With both tools
# - With custom state & custom context schemas
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,   # persistent state across calls
    context_schema=CustomContext,  # per-call contextual metadata
)


# -----------------------------------------------------------
# INVOKE AGENT
# The agent receives:
# - a user message: "greet the user"
# - context: user_id="user_123"
#
# Expected behavior:
# 1. Agent realizes it needs info → calls update_user_info()
# 2. update_user_info sets user_name = "John Smith"
# 3. Agent then knows it can call greet()
# 4. greet() returns "Hello John Smith!"
# -----------------------------------------------------------
agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/06_short_term_memory/main6_tool_writing_memory.py 


📦 PostgreSQL memory initialized
🤖 Tool Writing Memory Example
Tools can update agent memory state
Try: 'set my name to John', 'save preference theme dark', 'add note I like Python'

You: set my name Erkan
Agent: OK. I have set your name to Erkan.
You: save preference theme Blues
Agent: OK. I have saved your theme preference as Blues.
You: add note I like mercedes E180
Agent: OK. I've added the note "I like mercedes E180".
You: show me notes
Agent: Here are your notes:
- I like Blues
- I like mercedes E180
You: what is my name
Agent: Your name is Erkan.
You: quit
```