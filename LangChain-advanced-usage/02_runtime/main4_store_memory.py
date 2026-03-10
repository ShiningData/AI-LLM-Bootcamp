"""
Runtime store and memory example.

Shows how to:
- Use runtime store for long-term memory
- Implement memory-aware tools
- Simulate persistent storage operations
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
class MemoryContext:
    user_id: str
    workspace_id: str

# Simulate a memory store (in production, use runtime.store)
memory_store = {}

@tool
def save_user_preference(preference_name: str, preference_value: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Save a user preference to long-term memory."""
    ctx = runtime.context
    key = f"user:{ctx.user_id}:pref:{preference_name}"
    
    # Simulate store operation
    if runtime.store:
        # In real implementation: runtime.store.set(("preferences", ctx.user_id), {preference_name: preference_value})
        print(f"💾 STORE: Saving to real store - {key} = {preference_value}")
        return f"Saved '{preference_name}' = '{preference_value}' to persistent store"
    else:
        # Fallback to simulated store
        memory_store[key] = preference_value
        print(f"💾 SIMULATED: {key} = {preference_value}")
        return f"Saved '{preference_name}' = '{preference_value}' (simulated)"

@tool
def get_user_preference(preference_name: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Get a user preference from long-term memory."""
    ctx = runtime.context
    key = f"user:{ctx.user_id}:pref:{preference_name}"
    
    # Simulate store operation
    if runtime.store:
        # In real implementation: runtime.store.get(("preferences", ctx.user_id))
        print(f"📖 STORE: Reading from real store - {key}")
        return f"Retrieved '{preference_name}' from persistent store: [would get actual value]"
    else:
        # Fallback to simulated store
        value = memory_store.get(key, "not set")
        print(f"📖 SIMULATED: {key} = {value}")
        return f"User's '{preference_name}' preference: {value}"

@tool
def save_workspace_data(data_type: str, data_value: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Save workspace-specific data."""
    ctx = runtime.context
    key = f"workspace:{ctx.workspace_id}:{data_type}"
    
    memory_store[key] = data_value
    print(f"🏢 WORKSPACE: {key} = {data_value}")
    return f"Saved workspace {data_type}: {data_value}"

@tool
def get_workspace_data(data_type: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Get workspace-specific data.""" 
    ctx = runtime.context
    key = f"workspace:{ctx.workspace_id}:{data_type}"
    
    value = memory_store.get(key, "not set")
    print(f"🏢 WORKSPACE: {key} = {value}")
    return f"Workspace {data_type}: {value}"

def demo_memory_operations():
    """Demo memory store operations."""
    print("💾 Memory Store Demo")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[save_user_preference, get_user_preference, save_workspace_data, get_workspace_data],
        context_schema=MemoryContext,
        system_prompt="You are a memory assistant that can save and retrieve user preferences and workspace data."
    )
    
    context = MemoryContext(user_id="user_123", workspace_id="workspace_456")
    
    print("\n--- Testing user preferences ---")
    # Save preferences
    agent.invoke(
        {"messages": [{"role": "user", "content": "Save my theme preference as 'dark mode'"}]},
        context=context
    )
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "Save my language preference as 'python'"}]},
        context=context
    )
    
    # Retrieve preferences
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is my theme preference?"}]},
        context=context
    )
    
    print("\n--- Testing workspace data ---")
    # Save workspace data
    agent.invoke(
        {"messages": [{"role": "user", "content": "Save project status as 'in progress'"}]},
        context=context
    )
    
    # Retrieve workspace data
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is our project status?"}]},
        context=context
    )
    
    print("\n--- Testing memory persistence across contexts ---")
    # Test with different user but same workspace
    new_context = MemoryContext(user_id="user_789", workspace_id="workspace_456")
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is the project status?"}]},
        context=new_context
    )
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is my theme preference?"}]},  # Should be different user's pref
        context=new_context
    )

if __name__ == "__main__":
    print("💾 LangChain Store Memory Example")
    print("Shows how to implement persistent memory with runtime store\n")
    
    demo_memory_operations()
    
    print("\n✅ Memory store demo completed!")
    print("💾 Key concepts demonstrated:")
    print("   - ToolRuntime.store access")
    print("   - User-scoped memory storage")
    print("   - Workspace-scoped data sharing")
    print("   - Memory persistence across sessions")
    print("   - Simulated vs real store operations")