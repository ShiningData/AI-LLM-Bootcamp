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
def multipy(x: int, y: int):
    """Multiplies two integers"""
    return x * y

@tool
def add(x: int, y: int):
    """Adds two integers"""
    return x + y

tools = [multipy, add]


agent = create_agent(model=model,
             tools=tools,
             )

result = agent.invoke({"messages": [{"role": "user", "content": "What is 5 times 7?"}]})

print("Final answer:", result["messages"][-2].content)