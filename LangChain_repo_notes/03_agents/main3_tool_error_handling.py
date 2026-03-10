from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")


@tool
def search(query: str) -> str:
    """Search for information."""
    # Fake error to demonstrate error handling
    raise ValueError("Search service is temporarily unavailable")

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 24°C"


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)

human_message = HumanMessage(content="Find the most popular wireless headphones right now and check if they're in stock")

result = agent.invoke({"messages": [human_message]})
print("Final answer:", result["messages"][-1].content)

# Output
# Final answer: I'm sorry, I wasn't able to find the most popular wireless headphones because the search service is temporarily unavailable. 
# Please try again later.


"""
1. The handle_tool_errors middleware catches the ValueError
  2. It creates a ToolMessage with a helpful error message: "Tool error: Please check your input and try again. (Search service is
  temporarily unavailable)"
  3. The agent receives this error message and continues processing
  4. The agent responds appropriately: "I am sorry, I cannot fulfill this request. The search service is currently unavailable. Please
  try again later."

  Without the error handling middleware, the agent would have crashed with an unhandled exception. With it, the agent gracefully handles
   the error and provides a user-friendly response.
"""