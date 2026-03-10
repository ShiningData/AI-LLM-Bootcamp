"""
Tool context writing using Commands to update agent state.

Shows how tools can:
- Update authentication status in state
- Modify user preferences in store
- Set session variables and flags
- Persist changes across conversation turns
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.command import Command
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
    session_type: str

# Tools that write to different context sources using Commands

# 1. Tools writing to STATE using Commands
@tool
def authenticate_user(password: str, runtime: ToolRuntime[UserContext]) -> Command:
    """Authenticate user and update state."""
    user_id = runtime.context.user_id
    
    # Simple authentication logic
    valid_passwords = {"admin": "admin123", "user": "user123", "guest": "guest123"}
    
    if user_id in valid_passwords and password == valid_passwords[user_id]:
        # Write to state: mark as authenticated
        return Command(
            update={
                "authenticated": True,
                "login_time": "2024-01-15 14:30",
                "session_id": f"sess_{user_id}_001",
                "permissions": ["read", "write"] if user_id == "admin" else ["read"]
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "failed_attempts": runtime.state.get("failed_attempts", 0) + 1
            }
        )

@tool
def logout_user(runtime: ToolRuntime) -> Command:
    """Logout user and clear authentication state."""
    return Command(
        update={
            "authenticated": False,
            "session_id": None,
            "login_time": None,
            "permissions": [],
            "logout_time": "2024-01-15 16:45"
        }
    )

@tool
def set_user_permissions(permissions: str, runtime: ToolRuntime) -> Command:
    """Update user permissions in state."""
    # Parse permissions string
    perm_list = [p.strip() for p in permissions.split(",")]
    
    # Validate permissions
    valid_perms = ["read", "write", "delete", "admin"]
    filtered_perms = [p for p in perm_list if p in valid_perms]
    
    return Command(
        update={
            "permissions": filtered_perms,
            "permissions_updated": "2024-01-15 15:00"
        }
    )

@tool
def update_session_data(key: str, value: str, runtime: ToolRuntime) -> Command:
    """Update arbitrary session data in state."""
    # Get current session data
    current_session = runtime.state.get("session_data", {})
    
    # Update with new key-value pair
    current_session[key] = value
    
    return Command(
        update={
            "session_data": current_session,
            "last_updated": "2024-01-15 15:30"
        }
    )

# 2. Tools writing to STORE
@tool
def save_user_preference(preference_key: str, preference_value: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save user preference to persistent store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            # Get existing preferences
            existing_prefs = store.get(("user_preferences",), user_id)
            prefs = existing_prefs.value if existing_prefs else {}
            
            # Update preference
            prefs[preference_key] = preference_value
            
            # Save back to store
            store.put(("user_preferences",), user_id, prefs)
            
            return f"✅ Saved preference: {preference_key} = {preference_value}"
        except Exception as e:
            return f"❌ Failed to save preference: {str(e)}"
    else:
        return "❌ No store configured"

@tool
def update_user_profile(name: str, email: str, runtime: ToolRuntime[UserContext]) -> str:
    """Update user profile in persistent store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            profile_data = {
                "name": name,
                "email": email,
                "updated_at": "2024-01-15 16:00"
            }
            
            store.put(("user_profiles",), user_id, profile_data)
            return f"✅ Updated profile for {name} ({email})"
        except Exception as e:
            return f"❌ Failed to update profile: {str(e)}"
    else:
        return "❌ No store configured"

@tool
def record_user_activity(activity: str, runtime: ToolRuntime[UserContext]) -> str:
    """Record user activity in persistent store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            # Get existing activity log
            existing_log = store.get(("activity_log",), user_id)
            activities = existing_log.value if existing_log else []
            
            # Add new activity
            activities.append({
                "activity": activity,
                "timestamp": "2024-01-15 16:15",
                "session_type": runtime.context.session_type
            })
            
            # Keep only last 10 activities
            activities = activities[-10:]
            
            store.put(("activity_log",), user_id, activities)
            return f"✅ Recorded activity: {activity}"
        except Exception as e:
            return f"❌ Failed to record activity: {str(e)}"
    else:
        return "❌ No store configured"

# 3. Combined state and store updates
@tool
def complete_user_setup(username: str, theme: str, runtime: ToolRuntime[UserContext]) -> Command:
    """Complete user setup by updating both state and store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    # Update store with user preferences
    if store:
        try:
            setup_data = {
                "username": username,
                "theme": theme,
                "setup_completed": True,
                "setup_date": "2024-01-15 17:00"
            }
            store.put(("user_setup",), user_id, setup_data)
        except:
            pass  # Continue even if store update fails
    
    # Update state to mark setup as complete
    return Command(
        update={
            "setup_completed": True,
            "username": username,
            "preferences": {"theme": theme},
            "onboarding_stage": "completed"
        }
    )

@tool
def start_secure_session(security_level: str, runtime: ToolRuntime[UserContext]) -> Command:
    """Start a secure session with enhanced state tracking."""
    user_id = runtime.context.user_id
    
    # Determine session capabilities based on security level
    if security_level == "high":
        capabilities = ["read", "write", "delete", "admin"]
        session_timeout = 3600  # 1 hour
    elif security_level == "medium":
        capabilities = ["read", "write"]
        session_timeout = 7200  # 2 hours
    else:
        capabilities = ["read"]
        session_timeout = 14400  # 4 hours
    
    return Command(
        update={
            "secure_session": True,
            "security_level": security_level,
            "session_capabilities": capabilities,
            "session_timeout": session_timeout,
            "secure_session_started": "2024-01-15 17:30"
        }
    )

def demo_state_writing():
    """Demo tools writing to agent state using Commands."""
    print("✍️ Tools Writing to Agent State")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[authenticate_user, logout_user, set_user_permissions, update_session_data],
        context_schema=UserContext,
        system_prompt="You are an assistant that can manage user authentication and session state."
    )
    
    context = UserContext(user_id="admin", session_type="interactive")
    
    # Test authentication
    print("\n--- Testing user authentication ---")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Authenticate with password admin123"}],
            "authenticated": False
        },
        context=context
    )
    
    # Test permission updates
    print("\n--- Testing permission updates ---")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Set my permissions to read, write, admin"}],
            "authenticated": True
        },
        context=context
    )
    
    # Test session data updates
    print("\n--- Testing session data updates ---")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Set session data: current_project = langchain_demo"}],
            "authenticated": True
        },
        context=context
    )

def demo_store_writing():
    """Demo tools writing to persistent store."""
    print("\n💾 Tools Writing to Persistent Store")
    print("=" * 50)
    
    store = InMemoryStore()
    
    agent = create_agent(
        model=model,
        tools=[save_user_preference, update_user_profile, record_user_activity],
        context_schema=UserContext,
        store=store,
        system_prompt="You are an assistant that can manage user preferences and profiles."
    )
    
    context = UserContext(user_id="user123", session_type="web")
    
    print("\n--- Saving user preferences ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Save my theme preference as dark mode"}]},
        context=context
    )
    
    print("\n--- Updating user profile ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Update my profile: name is John Doe, email is john@example.com"}]},
        context=context
    )
    
    print("\n--- Recording user activity ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Record that I completed the tutorial"}]},
        context=context
    )
    
    # Show what was stored
    print("\n--- Checking stored data ---")
    try:
        prefs = store.get(("user_preferences",), "user123")
        profile = store.get(("user_profiles",), "user123")
        activities = store.get(("activity_log",), "user123")
        
        print(f"Stored preferences: {prefs.value if prefs else 'None'}")
        print(f"Stored profile: {profile.value if profile else 'None'}")
        print(f"Stored activities: {activities.value if activities else 'None'}")
    except Exception as e:
        print(f"Error checking stored data: {e}")

def demo_combined_updates():
    """Demo tools that update both state and store."""
    print("\n🔄 Combined State and Store Updates")
    print("=" * 50)
    
    store = InMemoryStore()
    
    agent = create_agent(
        model=model,
        tools=[complete_user_setup, start_secure_session],
        context_schema=UserContext,
        store=store,
        system_prompt="You are an assistant that can manage comprehensive user setup."
    )
    
    context = UserContext(user_id="new_user", session_type="onboarding")
    
    print("\n--- Completing user setup ---")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Complete my setup with username JohnD and dark theme"}],
            "setup_completed": False
        },
        context=context
    )
    
    print("\n--- Starting secure session ---")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Start a high security session"}],
            "authenticated": True
        },
        context=context
    )

def demo_state_persistence():
    """Demo how state changes persist across conversation turns."""
    print("\n🔄 State Persistence Across Turns")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[authenticate_user, update_session_data],
        context_schema=UserContext,
        system_prompt="You can check and modify session state."
    )
    
    context = UserContext(user_id="user", session_type="persistent")
    config = {"configurable": {"thread_id": "persistence_demo"}}
    
    # First interaction: authenticate
    print("\n--- Turn 1: Authentication ---")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Authenticate with password user123"}],
            "authenticated": False
        },
        context=context,
        config=config
    )
    
    # Second interaction: update session data (should remember auth state)
    print("\n--- Turn 2: Update session data ---")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Set session data: workspace = project_alpha"}]
        },
        context=context,
        config=config
    )
    
    # Third interaction: check state (should have all previous updates)
    print("\n--- Turn 3: Check current state ---")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "What is my current session state?"}]
        },
        context=context,
        config=config
    )

if __name__ == "__main__":
    print("✍️ LangChain Tool Context Writing Example")
    print("Shows how tools can update state and store using Commands\n")
    
    demo_state_writing()
    demo_store_writing()
    demo_combined_updates()
    demo_state_persistence()
    
    print("\n✅ Tool context writing demo completed!")
    print("✍️ Key concepts demonstrated:")
    print("   - Using Command to update agent state")
    print("   - Writing to persistent store from tools")
    print("   - Combined state and store updates")
    print("   - State persistence across conversation turns")
    print("   - Authentication and session management")