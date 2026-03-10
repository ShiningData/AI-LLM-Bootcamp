"""
Tool context reading from state, store, and runtime context.

Shows how tools can:
- Read authentication status from agent state
- Access user preferences from store
- Use runtime context for configuration
- Make decisions based on available context
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
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
class SecurityContext:
    user_id: str
    security_level: str  # low, medium, high
    environment: str     # dev, staging, prod

# Tools that read from different context sources

# 1. Tools reading from STATE
@tool
def check_authentication_status(runtime: ToolRuntime) -> str:
    """Check if the user is currently authenticated by reading from state."""
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)
    
    if is_authenticated:
        return "✅ User is authenticated and can access protected resources"
    else:
        return "❌ User is not authenticated. Please login first"

@tool
def check_session_info(runtime: ToolRuntime) -> str:
    """Get current session information from state."""
    state = runtime.state
    
    session_id = state.get("session_id", "unknown")
    last_activity = state.get("last_activity", "never")
    permissions = state.get("permissions", [])
    
    return f"Session: {session_id}, Last activity: {last_activity}, Permissions: {permissions}"

@tool
def get_conversation_context(runtime: ToolRuntime) -> str:
    """Get conversation context and history length from state."""
    state = runtime.state
    messages = state.get("messages", [])
    
    message_count = len(messages)
    if message_count < 5:
        context_level = "early conversation"
    elif message_count < 15:
        context_level = "established conversation"
    else:
        context_level = "extended conversation"
    
    return f"Conversation context: {message_count} messages ({context_level})"

# 2. Tools reading from STORE
@tool
def get_user_preferences(runtime: ToolRuntime[SecurityContext]) -> str:
    """Get user preferences from persistent store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            # Try to get user preferences
            prefs = store.get(("user_preferences",), user_id)
            if prefs:
                return f"User preferences: {prefs.value}"
            else:
                return "No user preferences stored yet"
        except:
            return "Unable to access user preferences store"
    else:
        return "No store configured"

@tool
def get_security_settings(runtime: ToolRuntime[SecurityContext]) -> str:
    """Get security settings for user from store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            security_settings = store.get(("security_settings",), user_id)
            if security_settings:
                settings = security_settings.value
                return f"Security settings: MFA={settings.get('mfa_enabled', False)}, " \
                       f"Session timeout={settings.get('session_timeout', '30min')}"
            else:
                return "Default security settings applied (no custom settings found)"
        except:
            return "Unable to access security settings"
    else:
        return "No security store configured"

@tool
def check_user_history(runtime: ToolRuntime[SecurityContext]) -> str:
    """Check user's interaction history from store."""
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            history = store.get(("user_history",), user_id)
            if history:
                hist_data = history.value
                return f"User history: {hist_data.get('total_sessions', 0)} sessions, " \
                       f"last login: {hist_data.get('last_login', 'unknown')}"
            else:
                return "No user history found (new user)"
        except:
            return "Unable to access user history"
    else:
        return "No history store configured"

# 3. Tools reading from RUNTIME CONTEXT
@tool
def check_environment_config(runtime: ToolRuntime[SecurityContext]) -> str:
    """Check environment and security configuration from runtime context."""
    ctx = runtime.context
    
    env_info = f"Environment: {ctx.environment}"
    security_info = f"Security level: {ctx.security_level}"
    user_info = f"User ID: {ctx.user_id}"
    
    return f"{env_info}, {security_info}, {user_info}"

@tool
def validate_operation_permissions(operation: str, runtime: ToolRuntime[SecurityContext]) -> str:
    """Validate if an operation is allowed based on context."""
    ctx = runtime.context
    state = runtime.state
    
    # Check authentication from state
    is_authenticated = state.get("authenticated", False)
    if not is_authenticated:
        return f"❌ Operation '{operation}' denied: User not authenticated"
    
    # Check security level from context
    if ctx.security_level == "low" and operation in ["delete", "admin"]:
        return f"❌ Operation '{operation}' denied: Insufficient security level"
    
    # Check environment restrictions from context
    if ctx.environment == "prod" and operation == "test":
        return f"❌ Operation '{operation}' denied: Test operations not allowed in production"
    
    return f"✅ Operation '{operation}' approved for user {ctx.user_id} in {ctx.environment}"

# 4. Combined context reading tool
@tool
def comprehensive_context_check(runtime: ToolRuntime[SecurityContext]) -> str:
    """Get comprehensive context information from all sources."""
    ctx = runtime.context
    state = runtime.state
    store = runtime.store
    
    context_info = []
    
    # From runtime context
    context_info.append(f"🔧 Runtime: User {ctx.user_id}, {ctx.environment} environment")
    
    # From state
    auth_status = "authenticated" if state.get("authenticated", False) else "not authenticated"
    message_count = len(state.get("messages", []))
    context_info.append(f"📊 State: {auth_status}, {message_count} messages")
    
    # From store (if available)
    if store:
        try:
            prefs = store.get(("user_preferences",), ctx.user_id)
            if prefs:
                context_info.append(f"💾 Store: User preferences found")
            else:
                context_info.append(f"💾 Store: No preferences stored")
        except:
            context_info.append(f"💾 Store: Access error")
    else:
        context_info.append(f"💾 Store: Not configured")
    
    return "\n".join(context_info)

def demo_state_reading():
    """Demo tools reading from agent state."""
    print("📊 Tools Reading from Agent State")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[check_authentication_status, check_session_info, get_conversation_context],
        context_schema=SecurityContext,
        system_prompt="You are an assistant with state-aware tools."
    )
    
    context = SecurityContext(user_id="user123", security_level="medium", environment="dev")
    
    # Test with unauthenticated state
    print("\n--- Unauthenticated state ---")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Check my authentication status and session info"}],
            "authenticated": False,
            "session_id": "sess_001",
            "permissions": ["read"]
        },
        context=context
    )
    
    # Test with authenticated state
    print("\n--- Authenticated state ---")
    agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Message 1"},
                {"role": "user", "content": "Message 2"},
                {"role": "user", "content": "Check my status and conversation context"}
            ],
            "authenticated": True,
            "session_id": "sess_002", 
            "last_activity": "2024-01-15 10:30",
            "permissions": ["read", "write"]
        },
        context=context
    )

def demo_store_reading():
    """Demo tools reading from persistent store."""
    print("\n💾 Tools Reading from Store")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Set up test data in store
    store.put(("user_preferences",), "user456", {
        "theme": "dark",
        "language": "python",
        "notifications": True
    })
    
    store.put(("security_settings",), "user456", {
        "mfa_enabled": True,
        "session_timeout": "1hour"
    })
    
    store.put(("user_history",), "user456", {
        "total_sessions": 15,
        "last_login": "2024-01-15"
    })
    
    agent = create_agent(
        model=model,
        tools=[get_user_preferences, get_security_settings, check_user_history],
        context_schema=SecurityContext,
        store=store,
        system_prompt="You are an assistant with store-aware tools."
    )
    
    context = SecurityContext(user_id="user456", security_level="high", environment="prod")
    
    print("\n--- User with stored preferences ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Get my preferences, security settings, and history"}]},
        context=context
    )
    
    # Test with user who has no stored data
    print("\n--- New user with no stored data ---")
    new_context = SecurityContext(user_id="new_user", security_level="low", environment="dev")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Check what information you have about me"}]},
        context=new_context
    )

def demo_context_reading():
    """Demo tools reading from runtime context."""
    print("\n🔧 Tools Reading from Runtime Context")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[check_environment_config, validate_operation_permissions],
        context_schema=SecurityContext,
        system_prompt="You are an assistant with context-aware security tools."
    )
    
    # Low security user
    print("\n--- Low security user attempting admin operation ---")
    low_sec_context = SecurityContext(user_id="user_low", security_level="low", environment="dev")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Check config and validate admin operation"}],
            "authenticated": True
        },
        context=low_sec_context
    )
    
    # High security admin in production
    print("\n--- High security admin in production ---")
    admin_context = SecurityContext(user_id="admin", security_level="high", environment="prod")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Validate delete operation and check environment"}],
            "authenticated": True
        },
        context=admin_context
    )

def demo_comprehensive_context():
    """Demo tool reading from all context sources."""
    print("\n🎯 Comprehensive Context Reading")
    print("=" * 50)
    
    store = InMemoryStore()
    store.put(("user_preferences",), "power_user", {"role": "analyst", "access_level": "full"})
    
    agent = create_agent(
        model=model,
        tools=[comprehensive_context_check],
        context_schema=SecurityContext,
        store=store,
        system_prompt="You are an assistant with comprehensive context awareness."
    )
    
    context = SecurityContext(user_id="power_user", security_level="high", environment="staging")
    
    agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Previous interaction"},
                {"role": "user", "content": "Give me a complete context overview"}
            ],
            "authenticated": True,
            "session_id": "sess_comprehensive"
        },
        context=context
    )

if __name__ == "__main__":
    print("🔍 LangChain Tool Context Reading Example")
    print("Shows how tools can read from state, store, and runtime context\n")
    
    demo_state_reading()
    demo_store_reading()
    demo_context_reading()
    demo_comprehensive_context()
    
    print("\n✅ Tool context reading demo completed!")
    print("🔍 Key concepts demonstrated:")
    print("   - Reading authentication from agent state")
    print("   - Accessing user preferences from store")
    print("   - Using runtime context for configuration")
    print("   - Combined context reading strategies")
    print("   - State-based decision making in tools")