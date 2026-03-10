"""
Dynamic System Prompts based on conversation state and user context.

Shows how to:
- Create state-aware dynamic prompts
- Use store-based user preferences
- Implement context-aware prompting
- Adapt prompts to user roles and environments
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore
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
    user_id: str
    user_role: str
    deployment_env: str

@tool
def simple_search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': Found relevant information."

# 1. State-aware dynamic prompt based on conversation length
@dynamic_prompt
def conversation_length_prompt(request: ModelRequest) -> str:
    """Adapt system prompt based on conversation length."""
    message_count = len(request.messages)
    
    base = "You are a helpful assistant."
    
    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."
    elif message_count > 5:
        base += "\nKeep responses focused and relevant."
    else:
        base += "\nFeel free to provide detailed explanations."
    
    return base

# 2. Store-aware prompt using user preferences
@dynamic_prompt  
def preference_aware_prompt(request: ModelRequest) -> str:
    """Build prompt based on user preferences from store."""
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    
    base = "You are a helpful assistant."
    
    # Try to get user preferences from store
    user_prefs = None
    if store:
        try:
            user_prefs = store.get(("preferences",), user_id)
        except:
            pass  # No preferences stored yet
    
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."
        
        if "detail_level" in user_prefs.value:
            detail = user_prefs.value["detail_level"]
            base += f"\nProvide {detail} level of detail."
    
    return base

# 3. Context-aware prompt based on user role and environment
@dynamic_prompt
def role_environment_prompt(request: ModelRequest) -> str:
    """Adapt prompt based on user role and deployment environment."""
    ctx = request.runtime.context
    
    base = "You are a helpful assistant."
    
    # Adjust based on user role
    if ctx.user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif ctx.user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."
    elif ctx.user_role == "editor":
        base += "\nYou can read and edit content, but not perform administrative tasks."
    
    # Adjust based on environment
    if ctx.deployment_env == "production":
        base += "\nBe extra careful with any data modifications."
    elif ctx.deployment_env == "staging":
        base += "\nThis is a testing environment - you can be more experimental."
    
    return base

def demo_conversation_length_prompts():
    """Demo prompts that adapt to conversation length."""
    print("📏 Conversation Length-Based Prompts")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[simple_search],
        middleware=[conversation_length_prompt],
        context_schema=UserContext,
        system_prompt="Base prompt (replaced by dynamic)"
    )
    
    context = UserContext(user_id="user1", user_role="user", deployment_env="dev")
    
    # Short conversation
    print("\n--- Short conversation (detailed explanations) ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is AI?"}]},
        context=context
    )
    
    # Simulate longer conversation
    print("\n--- Simulating longer conversation (concise responses) ---")
    long_messages = [{"role": "user", "content": f"Question {i}"} for i in range(15)]
    agent.invoke(
        {"messages": long_messages + [{"role": "user", "content": "Tell me about machine learning"}]},
        context=context
    )

def demo_preference_based_prompts():
    """Demo prompts based on stored user preferences."""
    print("\n🎨 Preference-Based Dynamic Prompts")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Set up user preferences
    store.put(("preferences",), "user_tech", {
        "communication_style": "technical",
        "detail_level": "high"
    })
    
    store.put(("preferences",), "user_casual", {
        "communication_style": "casual",
        "detail_level": "medium"
    })
    
    agent = create_agent(
        model=model,
        tools=[simple_search],
        middleware=[preference_aware_prompt],
        context_schema=UserContext,
        store=store,
        system_prompt="Base prompt (replaced by dynamic)"
    )
    
    # Technical user
    print("\n--- Technical user preferences ---")
    tech_context = UserContext(user_id="user_tech", user_role="user", deployment_env="dev")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Explain neural networks"}]},
        context=tech_context
    )
    
    # Casual user  
    print("\n--- Casual user preferences ---")
    casual_context = UserContext(user_id="user_casual", user_role="user", deployment_env="dev")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Explain neural networks"}]},
        context=casual_context
    )

def demo_role_environment_prompts():
    """Demo prompts that adapt to user roles and environments."""
    print("\n🔐 Role & Environment-Based Prompts")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[simple_search],
        middleware=[role_environment_prompt],
        context_schema=UserContext,
        system_prompt="Base prompt (replaced by dynamic)"
    )
    
    # Admin in production
    print("\n--- Admin user in production ---")
    admin_prod_context = UserContext(
        user_id="admin1", 
        user_role="admin", 
        deployment_env="production"
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me manage the database"}]},
        context=admin_prod_context
    )
    
    # Viewer in staging
    print("\n--- Viewer user in staging ---")
    viewer_staging_context = UserContext(
        user_id="viewer1",
        user_role="viewer", 
        deployment_env="staging"
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Show me the system status"}]},
        context=viewer_staging_context
    )

if __name__ == "__main__":
    print("🎯 LangChain Dynamic System Prompts Example")
    print("Shows how to adapt system prompts to conversation context\n")
    
    demo_conversation_length_prompts()
    demo_preference_based_prompts()
    demo_role_environment_prompts()
    
    print("\n✅ Dynamic prompts demo completed!")
    print("🎯 Key concepts demonstrated:")
    print("   - @dynamic_prompt decorator")
    print("   - State-aware prompt adaptation")
    print("   - Store-based user preferences")
    print("   - Role and environment context")
    print("   - ModelRequest.runtime access")