## Prompt
- Access short term memory (state) in middleware to create dynamic prompts based on conversation history or custom state fields.
- In this example you will see:
    - How to create custom context fields with TypedDict
    - How to use @dynamic_prompt middleware to generate personalized system prompts
    - How the dynamic prompt accesses runtime context to customize responses
    - How the agent uses the personalized prompt to address users by name
```python
from langchain.agents import create_agent
# create_agent: builds a LangChain v1-style agent.

from typing import TypedDict
# TypedDict: used for typed context input (lightweight alternative to Pydantic models).

from langchain.agents.middleware import dynamic_prompt, ModelRequest
# dynamic_prompt: decorator that lets you dynamically generate system prompts.
# ModelRequest: contains the request, runtime, messages, context, etc.


# -----------------------------------------------------------
# CUSTOM CONTEXT
# This is NOT persistent state.
# It is provided on each invocation and is used by middleware or tools.
# -----------------------------------------------------------
class CustomContext(TypedDict):
    user_name: str   # Passed per request; e.g., "John Smith"


# -----------------------------------------------------------
# SIMPLE TOOL
# The agent can call this tool if useful.
# -----------------------------------------------------------
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


# -----------------------------------------------------------
# DYNAMIC SYSTEM PROMPT MIDDLEWARE
# Runs BEFORE the LLM call.
# It generates a system prompt using the runtime context.
# -----------------------------------------------------------
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    # Read the per-request context field
    user_name = request.runtime.context["user_name"]
    
    # Create a customized system prompt based on user_name
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    
    return system_prompt
    # The agent replaces the default system prompt with this one.


# -----------------------------------------------------------
# CREATE THE AGENT
# - Uses dynamic system prompt middleware
# - Uses typed context
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5-nano",             # Main LLM
    tools=[get_weather],            # Weather tool available
    middleware=[dynamic_system_prompt],  # Attach dynamic system prompt middleware
    context_schema=CustomContext,   # Defines what context fields exist
)


# -----------------------------------------------------------
# INVOKE THE AGENT
# Provide:
# - a user message
# - context: user_name="John Smith"
#
# The dynamic prompt will run and generate a custom system prompt:
# "You are a helpful assistant. Address the user as John Smith."
# -----------------------------------------------------------
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)


# -----------------------------------------------------------
# PRINT ALL MESSAGES
# This will show:
# - the dynamic system prompt
# - user message
# - agent reasoning/tool calls
# - final answer addressing "John Smith"
# -----------------------------------------------------------
for msg in result["messages"]:
    msg.pretty_print()
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/06_short_term_memory/main7_dynamic_prompts.py 
```