# Long-term memory
- LangChain agents use LangGraph persistence to enable long-term memory. This is a more advanced topic and requires knowledge of LangGraph to use.

## Memory storage
- LangGraph stores long-term memories as JSON documents in a store.

- Each memory is organized under a custom namespace (similar to a folder) and a distinct key (like a file name). Namespaces often include user or org IDs or other labels that makes it easier to organize information.

- This structure enables hierarchical organization of memories. Cross-namespace searching is then supported through content filters.

- *how the LangGraph Store works*, *how memory is stored*, *how search works*, and *how namespaces isolate user data*.

- Using InMemoryStore for AI Memory + Semantic Search

```python
from langgraph.store.memory import InMemoryStore
# InMemoryStore → Key–value store with optional vector search.
# Used for:
#   - user memory
#   - context persistence
#   - storing preferences, traits, rules, summaries, embeddings
#   - retrieving items by metadata or semantic similarity
```

```python
# -----------------------------------------------------------
# SIMPLE EMBEDDING FUNCTION (for demonstration)
# -----------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    """
    Dummy embedding function.
    Replace with:
        - LangChain Embeddings
        - OpenAI / Claude / Gemini embeddings
        - BGE models
    """
    return [[1.0, 2.0] * len(texts)]
    # Produces repeatable but meaningless vectors (for demo only)
```

```python
# -----------------------------------------------------------
# INITIALIZE STORE WITH EMBEDDING SUPPORT
# -----------------------------------------------------------
store = InMemoryStore(
    index={
        "embed": embed,   # Embedding function used for semantic search
        "dims": 2         # Dimensionality of each embedding vector
    }
)
"""
InMemoryStore capabilities:
- put(): save memory
- get(): retrieve memory
- search(): vector similarity + metadata filter
- Namespaces isolate data per user or per conversation.

In production:
    - Use RedisStore, PostgresStore, or DynamoDBStore
    - Store embeddings + metadata persistently
"""
```

```python
# -----------------------------------------------------------
# NAMESPACE ORGANIZATION (USER → APP CONTEXT)
# -----------------------------------------------------------
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
"""
Namespaces are tuples:
    (user_id, application_context)

They isolate data so:
- each user has private memory
- each app mode has separate storage
(e.g., "coding", "chitchat", "task_management")
"""
```

```python
# -----------------------------------------------------------
# STORE A MEMORY ITEM
# -----------------------------------------------------------
store.put(
    namespace,         # Namespace to store under
    "a-memory",        # Unique ID for the memory
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",   # Any metadata field (useful for filtering)
    },
)
"""
put(namespace, key, value):
- Stores an item at: store[namespace][key] = value
- Value can include:
    - rules
    - traits
    - examples
    - metadata
    - anything JSON-serializable
"""
```

```python
# -----------------------------------------------------------
# RETRIEVE A MEMORY ITEM BY ID
# -----------------------------------------------------------
item = store.get(namespace, "a-memory")
"""
get(namespace, key):
- Direct dictionary-style lookup.
- Returns the exact stored memory object.
"""
```

```python
# -----------------------------------------------------------
# SEMANTIC SEARCH + METADATA FILTERING
# -----------------------------------------------------------
items = store.search(
    namespace,
    filter={"my-key": "my-value"},     # Must match metadata
    query="language preferences"       # Natural-language semantic search
)
"""
search(namespace, filter, query):
- Embeds the query text ("language preferences")
- Computes similarity against all embedded memories in namespace
- Filters results by metadata (my-key == "my-value")
- Returns items sorted by vector similarity (most relevant first)

This is how long-term memory retrieval works in LangGraph:
    - store many facts/preferences
    - retrieve based on semantic relevance
"""
```

### ✔️ 1. Memory storage & retrieval

Agents can store:

* user preferences
* conversation rules
* tone/style
* examples
* task instructions
* agent reflections

### ✔️ 2. Namespace isolation

Each user and each application context have separate memory spaces:

```
(my-user, chitchat)
(my-user, coding)
(another-user, chitchat)
```

### ✔️ 3. Semantic search over memory

The store can embed values and run similarity search.

This is what powers:

* personalized chatbots
* AI assistants with memory
* multi-step agents recalling past decisions
* agent reflections & improvements

### ✔️ 4. Production-ready architecture

Just replace `InMemoryStore` with:

* Postgres-based store
* Redis store
* DynamoDB store
* Weaviate / Qdrant vector store

## Read long-term memory in tools

- *how the store works*, *how tools read from it*, *how context passes user_id*, and *how `StoreValue` behaves*.

- Using InMemoryStore + ToolRuntime + Context

```python
from dataclasses import dataclass
# dataclass → clean way to define fields for agent runtime context

from langchain_core.runnables import RunnableConfig
# RunnableConfig → optional configuration for agent execution (not used directly here)

from langchain.agents import create_agent
# create_agent → Builds LangChain v1 agent (LLM + Tools + Middleware + Context + Store)

from langchain.tools import tool, ToolRuntime
# @tool → marks a function as a LangChain tool callable by the agent
# ToolRuntime → gives tool access to:
#     - runtime.state
#     - runtime.context
#     - runtime.store
#     - metadata like tool_call_id, model info, etc.

from langgraph.store.memory import InMemoryStore
# InMemoryStore → simple key-value store with optional vector search
# Ideal for demos; in production replace with RedisStore/PostgresStore/etc.
# -----------------------------------------------------------
# CONTEXT PASSED INTO THE AGENT
# -----------------------------------------------------------
@dataclass
class Context:
    user_id: str
"""
Context:
- This object is passed to agent.invoke(..., context=Context(...))
- Available inside tool via runtime.context
- Used here to identify which user's data to fetch from store
"""
# -----------------------------------------------------------
# INITIALIZE STORE AND INSERT USER DATA
# -----------------------------------------------------------
store = InMemoryStore() 
"""
InMemoryStore behaves like:
    store[(namespace)][key] = StoreValue(...)
Namespaces prevent data mixing across categories:
    ("users",)   → all user metadata
    ("memory",)  → agent long-term memory
    ("prefs",)   → preferences
"""

store.put(
    ("users",),       # namespace (tuple)
    "user_123",       # key (user ID)
    {
        "name": "John Smith",
        "language": "English",
    }                 # stored value (JSON-like dict)
)
"""
store.put(namespace, key, value)
- Saves structured data into the store.
- Wraps value in StoreValue object with metadata + embedding (if enabled).
"""
# -----------------------------------------------------------
# TOOL: READ USER INFO FROM STORE USING CONTEXT
# -----------------------------------------------------------
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """
    Look up user info from the store.
    Demonstrates:
        - runtime.store   → access to the same store passed to create_agent()
        - runtime.context → access to user_id provided during invocation
    """
    # Access store (same instance passed into create_agent)
    store = runtime.store

    # Pull user ID from execution context
    user_id = runtime.context.user_id

    # Retrieve StoreValue object (value + metadata)
    user_info = store.get(("users",), user_id)

    # Return stored info if found
    return str(user_info.value) if user_info else "Unknown user"
# -----------------------------------------------------------
# CREATE AGENT WITH TOOL + STORE + CONTEXT
# -----------------------------------------------------------
agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # Main LLM handling tool calls + reasoning
    tools=[get_user_info],               # Register tool
    store=store,                         # Attach store to runtime → tools can access it
    context_schema=Context               # Enables agent to accept Context object on invoke
)
"""
Important:
- store=store makes Store accessible to all tools at runtime.
- context_schema=Context tells agent: "I expect a Context object when invoked."
"""
# -----------------------------------------------------------
# RUN AGENT WITH CONTEXT
# -----------------------------------------------------------
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123") 
)
"""
Execution flow:

1. LLM receives user message.
2. LLM decides to call `get_user_info`.
3. LangGraph invokes tool with ToolRuntime:
     runtime.context.user_id == "user_123"
     runtime.store == InMemoryStore instance
4. Tool fetches data from store under ("users",) namespace
     → returns {"name": "John Smith", "language": "English"}
5. LLM generates final response.

Result:
    "John Smith, English" (formatted by model)
"""

# You can access the store directly to get the value
store.get(("users",), "user_123").value
```

### ✔️ 1. **Context → passes user identity**

The agent knows which user's data to fetch.

### ✔️ 2. **Store → persistent, shareable memory**

Stores:

* user metadata
* preferences
* agent memory
* domain-specific data

### ✔️ 3. **Tools read from the store via ToolRuntime**

`runtime.store.get(...)` gives the tool access to data without passing it in the prompt.

### ✔️ 4. **Agent design: LLM handles reasoning, tools handle data**

This is the MCP/LangGraph pattern of *LLM as brain, store as database*.

### ✔️ 5. **Easy to extend**

You can create tools for:

* writing memory
* updating preferences
* long-term chat memory
* task history
* embeddings + vector search
