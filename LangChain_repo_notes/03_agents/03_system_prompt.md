## System prompt
- You can shape how your agent approaches tasks by providing a prompt. The system_prompt parameter can be provided as a string:
```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```
- When no system_prompt is provided, the agent will infer its task from the messages directly.

### Dynamic system prompt
- For more advanced use cases where you need to modify the system prompt based on runtime context or agent state, you can use middleware.
- main4_dynamic_prompt.py 
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", max_tokens=200)


@tool
def search(query: str) -> str:
    """Search for information."""
    # Fake error to demonstrate error handling
    raise ValueError("Search service is temporarily unavailable")

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 24°C"


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        prompt = f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        prompt = f"{base_prompt} Explain concepts simply and avoid jargon."
    else:
        prompt = base_prompt

    print("DYNAMIC SYSTEM PROMPT:", prompt)  # <— see this in console
    return prompt


agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[user_role_prompt]
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "beginner"}
)

print(result)
```

- Run
```bash
 uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main4_dynamic_prompt.py 
 ```

### Note
```
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
...
```
This function does run, and create_agent(...) uses its return value as the system prompt that gets sent to the LLM.

But:
```
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "beginner"}
)

print(result)
```

result is just the final messages (human + AI) that LangChain decides to expose. System messages (including ones created by dynamic_prompt) are not included in that returned structure, so you won’t see:

"You are a helpful assistant. Explain concepts simply and avoid jargon."


anywhere in result, even though it was sent to the model.