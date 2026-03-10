from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

# Sample database
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith", 
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

# Context schema - immutable data passed to agent
@dataclass
class UserContext:
    user_id: str
    session_id: str = "default"

# 1. ToolRuntime - Access state (mutable conversation data)
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the current conversation."""
    messages = runtime.state["messages"]
    
    human_msgs = sum(1 for m in messages if hasattr(m, 'type') and m.type == "human")
    ai_msgs = sum(1 for m in messages if hasattr(m, 'type') and m.type == "ai")
    
    return f"📊 Conversation has {human_msgs} user messages and {ai_msgs} AI responses"

@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference from conversation state."""
    # Access custom state fields
    preferences = runtime.state.get("user_preferences", {})
    value = preferences.get(pref_name, "Not set")
    return f"🔧 Preference '{pref_name}': {value}"

# 2. Context - Access immutable configuration data
@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information using context."""
    # Access immutable context data
    user_id = runtime.context.user_id
    session_id = runtime.context.session_id
    
    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (f"👤 Account Info (Session: {session_id}):\n"
                f"Name: {user['name']}\n"
                f"Type: {user['account_type']}\n" 
                f"Balance: ${user['balance']}")
    return "❌ User not found"

@tool
def set_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """Set a user preference in the conversation state."""
    # Modify mutable state
    if "user_preferences" not in runtime.state:
        runtime.state["user_preferences"] = {}
    
    runtime.state["user_preferences"][key] = value
    return f"✅ Set preference '{key}' = '{value}'"

def main():
    print("🛠️ ToolRuntime & Context Demo")
    print("="*40)
    
    # Create agent with context schema
    agent = create_agent(
        model=model,
        tools=[summarize_conversation, get_user_preference, get_account_info, set_preference],
        context_schema=UserContext,
        system_prompt="You are a helpful assistant with access to user context and preferences."
    )
    
    # Choose user context
    print("Available users:")
    for uid, user in USER_DATABASE.items():
        print(f"  {uid}: {user['name']}")
    
    user_choice = input("\nEnter user ID (or press Enter for user123): ").strip()
    if not user_choice:
        user_choice = "user123"
    
    context = UserContext(user_id=user_choice, session_id="demo_session")
    
    print(f"\n🔑 Using context: user_id={context.user_id}, session_id={context.session_id}")
    print("\nTry these commands:")
    print("- 'What's my account info?'")
    print("- 'Set my preference theme to dark'") 
    print("- 'What's my theme preference?'")
    print("- 'Summarize our conversation'")
    print("- 'quit' to exit\n")
    
    # Initial state with some preferences
    state = {
        "messages": [],
        "user_preferences": {"language": "english", "notifications": "enabled"}
    }
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        state["messages"].append({"role": "user", "content": user_input})
        
        try:
            # Pass both state AND context
            result = agent.invoke(state, context=context)
            
            # Update state with new messages
            state = result
            
            response = result["messages"][-1].content
            if isinstance(response, list) and len(response) > 0:
                response = response[0].get('text', str(response))
                
            print(f"Agent: {response}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()