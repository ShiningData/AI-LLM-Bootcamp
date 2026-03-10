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