"""
Dynamic tool selection based on state, store, and runtime context.

Shows how to:
- Filter tools based on authentication state
- Enable/disable tools based on conversation progress
- Restrict tools based on user permissions
- Adapt toolset dynamically during conversation
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore
from typing import Callable
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
    department: str

# Define various tools with different access levels
@tool
def public_search(query: str) -> str:
    """Search public information (available to everyone)."""
    return f"Public search results for '{query}': Found general information."

@tool
def internal_search(query: str) -> str:
    """Search internal company database (requires authentication)."""
    return f"Internal search results for '{query}': Found company-specific data."

@tool
def admin_console(action: str) -> str:
    """Access admin console (admin only)."""
    return f"Admin action '{action}' executed successfully."

@tool
def hr_database(query: str) -> str:
    """Access HR database (HR department only)."""
    return f"HR database query '{query}': Found employee records."

@tool
def finance_reports(report_type: str) -> str:
    """Generate finance reports (finance department only)."""
    return f"Finance report '{report_type}' generated successfully."

@tool
def advanced_analytics(dataset: str) -> str:
    """Advanced analytics tool (enabled after trust is established)."""
    return f"Advanced analytics on '{dataset}': Generated insights and predictions."

# 1. State-based tool filtering
@wrap_model_call
def state_based_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation state and authentication."""
    
    state = request.state
    
    # Check authentication status
    is_authenticated = state.get("authenticated", False)
    
    # Count conversation progress
    message_count = len(state.get("messages", []))
    
    available_tools = []
    
    for tool in request.tools:
        tool_name = tool.name
        
        # Public tools are always available
        if tool_name.startswith("public_"):
            available_tools.append(tool)
        
        # Internal tools require authentication
        elif tool_name.startswith("internal_") and is_authenticated:
            available_tools.append(tool)
        
        # Advanced tools require established conversation
        elif tool_name.startswith("advanced_") and is_authenticated and message_count >= 5:
            available_tools.append(tool)
        
        # Admin tools require admin role
        elif tool_name.startswith("admin_"):
            user_role = request.runtime.context.user_role
            if user_role == "admin":
                available_tools.append(tool)
    
    if len(available_tools) != len(request.tools):
        print(f"🔧 TOOLS: Filtered from {len(request.tools)} to {len(available_tools)} tools")
        request = request.override(tools=available_tools)
    
    return handler(request)

# 2. Role and department-based tool filtering
@wrap_model_call
def role_based_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on user role and department."""
    
    ctx = request.runtime.context
    user_role = ctx.user_role
    department = ctx.department
    
    available_tools = []
    
    for tool in request.tools:
        tool_name = tool.name
        include_tool = False
        
        # Public tools for everyone
        if tool_name.startswith("public_"):
            include_tool = True
        
        # Internal tools for authenticated users
        elif tool_name.startswith("internal_"):
            if user_role in ["user", "manager", "admin"]:
                include_tool = True
        
        # Admin tools for admins only
        elif tool_name.startswith("admin_"):
            if user_role == "admin":
                include_tool = True
        
        # Department-specific tools
        elif tool_name.startswith("hr_"):
            if department == "hr" or user_role == "admin":
                include_tool = True
        
        elif tool_name.startswith("finance_"):
            if department == "finance" or user_role == "admin":
                include_tool = True
        
        # Advanced tools for managers and above
        elif tool_name.startswith("advanced_"):
            if user_role in ["manager", "admin"]:
                include_tool = True
        
        if include_tool:
            available_tools.append(tool)
    
    if len(available_tools) != len(request.tools):
        print(f"🔐 RBAC: User {ctx.user_id} ({user_role}, {department}) has access to {len(available_tools)}/{len(request.tools)} tools")
        request = request.override(tools=available_tools)
    
    return handler(request)

# 3. Store-based tool permissions
@wrap_model_call
def permission_based_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on stored user permissions."""
    
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    
    if store:
        try:
            # Get user permissions from store
            user_permissions = store.get(("permissions",), user_id)
            
            if user_permissions:
                allowed_tools = user_permissions.value.get("allowed_tools", [])
                
                # Filter tools based on stored permissions
                available_tools = [
                    tool for tool in request.tools
                    if tool.name in allowed_tools or tool.name.startswith("public_")
                ]
                
                if len(available_tools) != len(request.tools):
                    print(f"📋 PERMISSIONS: User {user_id} permissions limit tools to {len(available_tools)}")
                    request = request.override(tools=available_tools)
        except:
            pass  # No permissions stored, use default filtering
    
    return handler(request)

def demo_state_based_filtering():
    """Demo tool filtering based on authentication and conversation state."""
    print("🔒 State-Based Tool Filtering")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[public_search, internal_search, advanced_analytics],
        middleware=[state_based_tool_filter],
        context_schema=UserContext,
        system_prompt="You are an assistant with state-based tool access."
    )
    
    context = UserContext(user_id="user1", user_role="user", department="engineering")
    
    # Unauthenticated user
    print("\n--- Unauthenticated user (public tools only) ---")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Search for information about AI"}],
            "authenticated": False
        },
        context=context
    )
    
    # Authenticated user, short conversation
    print("\n--- Authenticated user, early conversation ---")
    agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Message 1"},
                {"role": "user", "content": "Search internal docs and run analytics"}
            ],
            "authenticated": True
        },
        context=context
    )
    
    # Authenticated user, established conversation
    print("\n--- Authenticated user, established conversation ---")
    long_messages = [{"role": "user", "content": f"Message {i}"} for i in range(6)]
    agent.invoke(
        {
            "messages": long_messages + [{"role": "user", "content": "Run advanced analytics"}],
            "authenticated": True
        },
        context=context
    )

def demo_role_based_filtering():
    """Demo tool filtering based on user roles and departments."""
    print("\n👥 Role & Department-Based Tool Filtering")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[public_search, internal_search, admin_console, hr_database, finance_reports],
        middleware=[role_based_tool_filter],
        context_schema=UserContext,
        system_prompt="You are an assistant with role-based tool access."
    )
    
    # Regular user
    print("\n--- Regular user (limited access) ---")
    user_context = UserContext(user_id="john", user_role="user", department="engineering")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me search and access admin tools"}]},
        context=user_context
    )
    
    # HR manager
    print("\n--- HR manager (HR tools access) ---")
    hr_context = UserContext(user_id="sarah", user_role="manager", department="hr")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Search employee records and generate reports"}]},
        context=hr_context
    )
    
    # Admin user
    print("\n--- Admin user (full access) ---")
    admin_context = UserContext(user_id="admin", user_role="admin", department="it")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Use admin console and access all systems"}]},
        context=admin_context
    )

def demo_permission_based_filtering():
    """Demo tool filtering based on stored permissions."""
    print("\n📋 Permission-Based Tool Filtering")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Set up user permissions
    store.put(("permissions",), "limited_user", {
        "allowed_tools": ["public_search", "internal_search"]
    })
    
    store.put(("permissions",), "power_user", {
        "allowed_tools": ["public_search", "internal_search", "hr_database", "advanced_analytics"]
    })
    
    agent = create_agent(
        model=model,
        tools=[public_search, internal_search, hr_database, finance_reports, advanced_analytics],
        middleware=[permission_based_tool_filter],
        context_schema=UserContext,
        store=store,
        system_prompt="You are an assistant with permission-based tool access."
    )
    
    # Limited user
    print("\n--- Limited user permissions ---")
    limited_context = UserContext(user_id="limited_user", user_role="user", department="support")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me search and access databases"}]},
        context=limited_context
    )
    
    # Power user
    print("\n--- Power user permissions ---")
    power_context = UserContext(user_id="power_user", user_role="analyst", department="data")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Use all available tools for analysis"}]},
        context=power_context
    )

if __name__ == "__main__":
    print("🔧 LangChain Dynamic Tool Selection Example")
    print("Shows how to filter and control tool access dynamically\n")
    
    demo_state_based_filtering()
    demo_role_based_filtering()
    demo_permission_based_filtering()
    
    print("\n✅ Dynamic tool selection demo completed!")
    print("🔧 Key concepts demonstrated:")
    print("   - State-based authentication filtering")
    print("   - Role-based access control (RBAC)")
    print("   - Department-specific tool access")
    print("   - Store-based permission management")
    print("   - request.override(tools=...) modification")