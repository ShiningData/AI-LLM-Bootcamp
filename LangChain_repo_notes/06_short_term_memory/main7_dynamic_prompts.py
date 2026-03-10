"""
Dynamic prompts example with middleware.

In this example you will see:
- How to create custom context fields with TypedDict
- How to use @dynamic_prompt middleware to generate personalized system prompts
- How the dynamic prompt accesses runtime context to customize responses
- How the agent uses the personalized prompt to address users by name
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

class CustomContext(TypedDict):
    user_name: str

def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)

for msg in result["messages"]:
    msg.pretty_print()