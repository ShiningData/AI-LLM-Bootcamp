## Short-term memory
### Overview
- Memory is a system that remembers information about previous interactions. For AI agents, memory is crucial because it lets them remember previous interactions, learn from feedback, and adapt to user preferences. As agents tackle more complex tasks with numerous user interactions, this capability becomes essential for both efficiency and user satisfaction.
Short term memory lets your application remember previous interactions within a single thread or conversation.

- Conversation history is the most common form of short-term memory. Long conversations pose a challenge to today’s LLMs; a full history may not fit inside an LLM’s context window, resulting in an context loss or errors.
**Even if your model supports the full context length, most LLMs still perform poorly over long contexts.** They get **“distracted”** by stale or off-topic content, all while suffering from slower response times and higher costs.
Chat models accept context using messages, which include instructions (a system message) and inputs (human messages). In chat applications, messages alternate between human inputs and model responses, resulting in a list of messages that grows longer over time. Because context windows are limited, many applications can benefit from using techniques to remove or “forget” stale information.

- To add short-term memory (thread-level persistence) to an agent, you need to specify a checkpointer when creating an agent.
`pip install langgraph-checkpoint-postgres`

```python
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver  


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        [get_user_info],
        checkpointer=checkpointer,  
    )
```

### Customizing agent memory
- By default, agents use AgentState to manage short term memory, specifically the conversation history via a messages key.
- You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the state_schema parameter.
- Custom state fields like user_id and preferences are serialized and stored within the existing checkpoint data structure in PostgreSQL. The database schema remains the same regardless of what custom fields you add to your AgentState class. The custom state data is simply stored as part of the serialized state blob in the established checkpoint tables.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver  

class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        state_schema=CustomAgentState, 
        [get_user_info],
        checkpointer=checkpointer,  
    )

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  
        "preferences": {"theme": "dark"}  
    },
    {"configurable": {"thread_id": "1"}})
```

### What is the difference checkpointer=InMemorySaver() and store=nMemoryStore() in create_agent?
- They serve different purposes:

  - checkpointer=InMemorySaver():
    - Saves entire conversation state (messages, agent state, etc.)
    - Used for resuming conversations from where you left off
    - Saves everything the agent needs to continue a conversation
    - Each checkpoint = snapshot of the whole conversation at a point in time

  - store=store:
    - Saves specific data that tools put/get
    - Used for cross-conversation memory (facts, user profiles, etc.)
    - Tools explicitly save/retrieve data using runtime.store
    - More like a database that tools can access

- Example:

  - Checkpointer - saves conversation state
  agent = create_agent(
      model=model,
      checkpointer=InMemorySaver(),  # Can resume conversation later
  )

 - Store - tools can save/get data  
  agent = create_agent(
      model=model,
      store=InMemoryStore(),  # Tools can save user profiles, facts, etc.
      tools=[save_user_info, get_user_info]
  )

  - Both together
  ```python
  agent = create_agent(
      model=model,
      checkpointer=InMemorySaver(),  # Resume conversations
      store=InMemoryStore(),         # Tools save data
      tools=[save_user_info, get_user_info]
  )
    ```
  Think of it as:
  - Checkpointer = Save/load entire conversation
  - Store = Database that tools can write to/read from