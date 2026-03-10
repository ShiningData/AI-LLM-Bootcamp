from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    # max_retries, api_key

    )

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's snowy in {location}."


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What is the capital of France?")

print(response)