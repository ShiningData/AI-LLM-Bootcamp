from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()


model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Yu have searched this: {query}"

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model=model,
    tools=[search],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567 he lives in USA California but sometimes visit Japan"}]
})

print(result["structured_response"])