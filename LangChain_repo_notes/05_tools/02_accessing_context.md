## Accessing Context
- Tools are most powerful when they can access agent state, runtime context, and long-term memory. This enables tools to make context-aware decisions, personalize responses, and maintain information across conversations.
- Runtime context provides a way to inject dependencies (like database connections, user IDs, or configuration) into your tools at runtime, making them more testable and reusable.
- Tools can access runtime information through the ToolRuntime parameter, which provides:
    - State - Mutable data that flows through execution (e.g., messages, counters, custom fields)
    - Context - Immutable configuration like user IDs, session details, or application-specific configuration
    - Store - Persistent long-term memory across conversations
    - Stream Writer - Stream custom updates as tools execute
    - Config - RunnableConfig for the execution
    - Tool Call ID - ID of the current tool call

### ToolRuntime
- Use ToolRuntime to access all runtime information in a single parameter. Simply add runtime: ToolRuntime to your tool signature, and it will be automatically injected without being exposed to the LLM.
- ToolRuntime: A unified parameter that provides tools access to state, context, store, streaming, config, and tool call ID. 
- The tool_runtime parameter is hidden from the model. For the example below, the model only sees pref_name in the tool schema - tool_runtime is not included in the request.
- Accessing state:
Tools can access the current graph state using ToolRuntime:
```python
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

#### Context
- Access immutable configuration and contextual data like user IDs, session details, or application-specific configuration through runtime.context.
Tools can access runtime context through ToolRuntime:
```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
```

#### Example
- Key concepts demonstrated:
  1. ToolRuntime.state (mutable):
    - summarize_conversation - reads conversation messages
    - set_preference - modifies user preferences in state
  2. ToolRuntime.context (immutable):
    - get_account_info - accesses user_id from context
    - Shows how context is passed separately from state

  The difference:
  - State = changes during conversation (messages, preferences)
  - Context = fixed for the session (user_id, session_id)

  Run it to see how tools can access both mutable conversation data and immutable configuration!

```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/05_tools/main3_context_example.py
```
#####  Example Dialogs
- Example Dialog 1: Context (Immutable Data)
  You: What's my account info?
  Agent: 👤 Account Info (Session: demo_session):
  Name: Bob Smith
  Type: Standard
  Balance: $1200
  This shows how the tool accessed immutable context data (user_id=user456) to look up account info

- Example Dialog 2: State (Mutable Data)
  You: Set my preference theme to dark
  Agent: ✅ Set preference 'theme' = 'dark'

  You: What's my theme preference?
  Agent: 🔧 Preference 'theme': dark
  This shows how tools can modify and read from conversation state

- Example Dialog 3: State Evolution
  You: Summarize our conversation
  Agent: 📊 Conversation has 4 user messages and 4 AI responses

  You: Set notification preference to off
  Agent: ✅ Set preference 'notification' = 'off'

  You: Summarize our conversation
  Agent: 📊 Conversation has 6 user messages and 6 AI responses
  This shows how state changes as the conversation progresses

- The Key Difference:
  - Context (user456, session_id) → Never changes during this conversation
  - State (messages, preferences) → Grows and changes with each interaction


#### Memory (Store)
Access persistent data across conversations using the store. The store is accessed via runtime.store and allows you to save and retrieve user-specific or application-specific data.
Tools can access and update the store through ToolRuntime:
```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
```

##### Example Dialogs for Memory Store

**First Conversation Session:**
```
You: Save user abc123 named Alice age 28 email alice@company.com
Agent: ✅ Saved user info for abc123 to persistent store

You: List all stored users
Agent: 📋 Stored Users:
- abc123: Alice
```

**Second Conversation Session (restart the program):**
```
You: Get user info for abc123
Agent: 💾 Stored User Info:
ID: abc123
Name: Alice
Age: 28
Email: alice@company.com

You: Save user xyz789 named Bob age 35 email bob@test.com
Agent: ✅ Saved user info for xyz789 to persistent store

You: List all stored users  
Agent: 📋 Stored Users:
- abc123: Alice
- xyz789: Bob
```

**Key Concept:**
- **Store** = Survives across program restarts and conversation sessions
- **State** = Only lasts for current conversation
- **Context** = Fixed for current session

The store demonstrates TRUE persistence - data saved in one conversation is available in completely separate conversation sessions!

```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/05_tools/store_example.py
```

#### Stream Writer
Stream custom updates from tools as they execute using runtime.stream_writer. This is useful for providing real-time feedback to users about what a tool is doing.
```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```
- If you use runtime.stream_writer inside your tool, the tool must be invoked within a LangGraph execution context. 
