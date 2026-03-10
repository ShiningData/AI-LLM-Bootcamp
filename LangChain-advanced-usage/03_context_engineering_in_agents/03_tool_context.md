# Tool Context
- Tools are special in that they both read and write context.
In the most basic case, when a tool executes, it receives the LLM’s request parameters and returns a tool message back. The tool does its work and produces a result.
- Tools can also fetch important information for the model that allows it to perform and complete tasks.
​
## 1. Reads
- Most real-world tools need more than just the LLM’s parameters. They need user IDs for database queries, API keys for external services, or current session state to make decisions. Tools read from state, store, and runtime context to access this information.

### 1.1. State
- Checking Authentication Using ToolRuntime

```python
from langchain.tools import tool, ToolRuntime
# @tool → declares a tool the agent can call
# ToolRuntime → provides access to:
#   - runtime.state (agent state)
#   - runtime.context (user context)
#   - runtime.store (persistent memory)
#   - tool_call_id, metadata, etc.

from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent that can call tools.


# -----------------------------------------------------------
# TOOL: CHECK AUTHENTICATION STATUS FROM STATE
# -----------------------------------------------------------
@tool
def check_authentication(
    runtime: ToolRuntime
) -> str:
    """
    Check whether the user is authenticated by examining
    the agent's current state.

    Tools can read (not write) agent state via runtime.state.
    """

    # Access the full state dictionary
    current_state = runtime.state

    # Read the "authenticated" flag (default False)
    is_authenticated = current_state.get("authenticated", False)

    # Return a natural-language response for the agent
    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"


# -----------------------------------------------------------
# BUILD AN AGENT THAT USES THE AUTH CHECK TOOL
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                   # Main LLM
    tools=[check_authentication]      # Tool available to the agent
)
```

---

#### 🔐 **Tools can inspect agent state**

Using:

```python
runtime.state
```

A tool can read:

* flags (e.g., `"authenticated": True`)
* values set by middleware or model
* progress markers
* permissions
* other stateful metadata

#### 🧠 **Agent state is a shared memory**

It is updated across the entire agent lifecycle:

* LLM outputs
* tool calls
* middleware updates

Thus, tools can make decisions such as:

* “User must authenticate before using this tool”
* “This tool is only available after step 5”
* “This action requires approval”

#### 🧩 **Lightweight authentication pattern**

For example:

```python
state["authenticated"] = True
```

…and tools can react accordingly.

#### ⚙️ No need for context_schema or stores

This tool uses only ephemeral state, not persistent memory or user context.

### 1.2. Store

### 1.3. Runtime Context