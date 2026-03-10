"""
Basic runtime context example with LangChain agents.

Shows how to:
- Define context schemas
- Access context in tools with ToolRuntime
- Pass context when invoking agents
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class UserContext:
    user_name: str
    user_id: str
    role: str

# Define tools that use runtime context
@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get user information from the runtime context."""
    ctx = runtime.context
    return f"User: {ctx.user_name} (ID: {ctx.user_id}, Role: {ctx.role})"

@tool  
def personalized_search(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search with personalized results based on user context."""
    user_name = runtime.context.user_name
    role = runtime.context.role
    
    # Customize search based on user role
    if role == "admin":
        scope = "system-wide"
    elif role == "manager": 
        scope = "department-wide"
    else:
        scope = "personal"
    
    return f"Search '{query}' for {user_name} ({scope} scope): Found relevant results."

def demo_basic_context():
    """Demo basic context access in tools."""
    print("🎯 Basic Runtime Context Demo")
    print("=" * 50)
    
    # Create agent with context schema
    agent = create_agent(
        model=model,
        tools=[get_user_info, personalized_search],
        context_schema=UserContext,
        system_prompt="You are a helpful assistant with access to user context."
    )
    
    # Test with different user contexts
    contexts = [
        UserContext(user_name="Alice", user_id="001", role="admin"),
        UserContext(user_name="Bob", user_id="002", role="user"),
        UserContext(user_name="Carol", user_id="003", role="manager")
    ]
    
    for ctx in contexts:
        print(f"\n--- Testing with {ctx.user_name} ({ctx.role}) ---")
        agent.invoke(
            {"messages": [{"role": "user", "content": "Get my info and search for project updates"}]},
            context=ctx
        )
        print(f"✅ Context successfully used for {ctx.user_name}")

if __name__ == "__main__":
    print("🏃 LangChain Basic Runtime Context Example")
    print("Shows how tools can access user context for personalization\n")
    
    demo_basic_context()
    
    print("\n✅ Basic context demo completed!")
    print("🎯 Key concepts demonstrated:")
    print("   - Context schema definition with @dataclass")
    print("   - ToolRuntime[Context] parameter in tools")
    print("   - Passing context when invoking agents")
    print("   - Role-based customization in tools")