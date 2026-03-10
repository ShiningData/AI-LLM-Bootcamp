## Runtime
- LangChain’s create_agent runs on LangGraph’s runtime under the hood.
- LangGraph exposes a Runtime object with the following information:
    - Context: static information like user id, db connections, or other dependencies for an agent invocation
    - Store: a BaseStore instance used for long-term memory
    - Stream writer: an object used for streaming information via the "custom" stream mode

- Runtime context provides dependency injection for your tools and middleware. Instead of hardcoding values or using global state, you can inject runtime dependencies (like database connections, user IDs, or configuration) when invoking your agent. This makes your tools more testable, reusable, and flexible.
- You can access the runtime information within tools and middleware.

### Access
When creating an agent with create_agent, you can specify a context_schema to define the structure of the context stored in the agent Runtime.
When invoking the agent, pass the context argument with the relevant configuration for the run:
```python
from dataclasses import dataclass

from langchain.agents import create_agent


@dataclass
class Context:
    user_name: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    context_schema=Context  
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")  
)
```
### Inside tools
You can access the runtime information inside tools to:
Access the context
Read or write long-term memory
Write to the custom stream (ex, tool progress / updates)
Use the ToolRuntime parameter to access the Runtime object inside a tool.
```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime  

@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:  
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id  

    preferences: str = "The user prefers you to write a brief and polite email."
    if runtime.store:  
        if memory := runtime.store.get(("users",), user_id):  
            preferences = memory.value["preferences"]

    return preferences
```
### Inside middleware
You can access runtime information in middleware to create dynamic prompts, modify messages, or control agent behavior based on user context.
Use request.runtime to access the Runtime object inside middleware decorators. The runtime object is available in the ModelRequest parameter passed to middleware functions.

```python
from dataclasses import dataclass
# dataclass → lightweight way to define structured context objects

from langchain.messages import AnyMessage
# AnyMessage → generic message type used by LangChain

from langchain.agents import create_agent, AgentState
# create_agent → builds the LangChain v1 agent
# AgentState   → holds messages + intermediate state

from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    before_model,
    after_model
)
# dynamic_prompt → dynamically generate system prompt per request
# before_model   → code executed BEFORE the LLM call
# after_model    → code executed AFTER the LLM call

from langgraph.runtime import Runtime
# Runtime → gives access to context + execution environment (LangGraph)


# -----------------------------------------------------------
# DEFINE CUSTOM CONTEXT PASSED TO THE AGENT
# -----------------------------------------------------------
@dataclass
class Context:
    user_name: str   # Custom per-request value available to middleware


# -----------------------------------------------------------
# 1) DYNAMIC SYSTEM PROMPT
# -----------------------------------------------------------
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """
    Generate the system prompt dynamically, using the request context.
    Runs before every LLM call.
    """
    user_name = request.runtime.context.user_name
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


# -----------------------------------------------------------
# 2) BEFORE MODEL HOOK
# -----------------------------------------------------------
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """
    Runs immediately BEFORE the LLM is called.
    Excellent for logging, instrumentation, tracing.
    """
    print(f"Processing request for user: {runtime.context.user_name}")
    return None   # No modification to state


# -----------------------------------------------------------
# 3) AFTER MODEL HOOK
# -----------------------------------------------------------
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """
    Runs immediately AFTER the LLM finishes producing output.
    Good place for auditing, debugging, analytics.
    """
    print(f"Completed request for user: {runtime.context.user_name}")
    return None   # No modification to state


# -----------------------------------------------------------
# BUILD THE AGENT
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",                         # The LLM that does the work
    tools=[...],                                # Tools the agent can use
    middleware=[                                 # Middleware pipeline
        dynamic_system_prompt,                  # Dynamic per-user system prompt
        log_before_model,                       # Logging before model call
        log_after_model,                        # Logging after model call
    ],
    context_schema=Context                      # Allow passing custom context object
)


# -----------------------------------------------------------
# INVOKE THE AGENT WITH CUSTOM CONTEXT
# -----------------------------------------------------------
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")     # Passed to ALL hooks and dynamic prompt
)
```
### Examples
1. main1_basic_context.py - Basic context access in tools
2. main2_dynamic_prompts.py - Dynamic prompt generation
3. main3_lifecycle_hooks.py  - Before/after model hooks
4. main4_store_memory.py - Memory store operations