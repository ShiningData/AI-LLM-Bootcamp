from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gemini-2.5-flash-lite",
    model_provider="google_genai",
    temperature=0.1,
    max_tokens=500,
)

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

tools = [get_weather]

agent = create_agent(
    model=model,
    tools=tools
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Boston?"}]})
print("Final answer:", result["messages"][-1].content)