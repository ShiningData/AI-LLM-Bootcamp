from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=200)


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