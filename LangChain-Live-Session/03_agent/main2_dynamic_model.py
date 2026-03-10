from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

basic_model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")
advanced_model = init_chat_model(model="gemini-2.5-pro", model_provider="google_genai")

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two integers"""
    return x * y

tools = [multiply]

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
        model_name = "gemini-2.5-pro"
    else:
        model = basic_model
        model_name = "gemini-2.5-flash-lite"
    
    print(f"Using model: {model_name} (message count: {message_count})")
    
    response = handler(request.override(model=model))
    
    # You can also check the response metadata
    if hasattr(response, 'response_metadata') and 'model_name' in response.response_metadata:
        print(f"Confirmed model used: {response.response_metadata['model_name']}")
    
    return response

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 5 times 7?"}]})
print("Final answer:", result["messages"][-1].content)