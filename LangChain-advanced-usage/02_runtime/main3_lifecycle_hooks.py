"""
Runtime lifecycle hooks example.

Shows how to:
- Use @before_model and @after_model decorators
- Access Runtime in middleware hooks
- Implement logging, auditing, and monitoring
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime
from dotenv import load_dotenv
import time

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class AuditContext:
    user_id: str
    session_id: str
    environment: str  # dev, staging, prod

# Global state for demo (in production, use proper logging/metrics systems)
request_metrics = {}

@before_model
def audit_request_start(state, runtime: Runtime[AuditContext]) -> dict:
    """Log and audit request initiation."""
    ctx = runtime.context
    timestamp = time.time()
    
    print(f"🔍 AUDIT: Request started")
    print(f"   User: {ctx.user_id}")
    print(f"   Session: {ctx.session_id}")
    print(f"   Environment: {ctx.environment}")
    print(f"   Time: {timestamp}")
    
    # Store metrics for later use
    request_metrics['start_time'] = timestamp
    request_metrics['user_id'] = ctx.user_id
    
    return None  # No state modification

@after_model  
def audit_request_complete(state, runtime: Runtime[AuditContext]) -> dict:
    """Log request completion and calculate metrics."""
    ctx = runtime.context
    end_time = time.time()
    
    # Calculate duration
    start_time = request_metrics.get('start_time', end_time)
    duration = end_time - start_time
    
    print(f"✅ AUDIT: Request completed")
    print(f"   Duration: {duration:.3f}s")
    print(f"   User: {ctx.user_id}")
    
    # In production: send to monitoring system
    print(f"📊 METRICS: Logged to monitoring dashboard")
    
    return None  # No state modification

@before_model
def security_check(state, runtime: Runtime[AuditContext]) -> dict:
    """Perform security checks before processing."""
    ctx = runtime.context
    
    # Simulate security validation
    if ctx.environment == "prod" and ctx.user_id.startswith("test_"):
        print("🚫 SECURITY: Test user blocked in production")
        raise ValueError("Test users not allowed in production")
    
    print(f"🔒 SECURITY: User {ctx.user_id} authorized for {ctx.environment}")
    return None

@tool
def secure_operation(operation: str, runtime: ToolRuntime[AuditContext]) -> str:
    """Perform a secure operation with full auditing."""
    ctx = runtime.context
    
    print(f"🔧 OPERATION: Executing '{operation}' for user {ctx.user_id}")
    return f"Operation '{operation}' completed successfully in {ctx.environment}"

def demo_lifecycle_hooks():
    """Demo before/after model hooks."""
    print("🔄 Lifecycle Hooks Demo")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[secure_operation],
        middleware=[
            security_check,      # Security before model
            audit_request_start, # Audit before model  
            audit_request_complete  # Audit after model
        ],
        context_schema=AuditContext,
        system_prompt="You are a secure assistant with full audit logging."
    )
    
    # Test valid request
    print("\n--- Testing valid request ---")
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "Perform a data backup operation"}]},
            context=AuditContext(
                user_id="admin_001",
                session_id="sess_123",
                environment="prod"
            )
        )
        print("✅ Valid request processed with full audit trail")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test blocked request
    print("\n--- Testing blocked request ---")
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "Test operation"}]},
            context=AuditContext(
                user_id="test_user",
                session_id="sess_456", 
                environment="prod"
            )
        )
        print("❌ Blocked request should not have succeeded")
    except Exception as e:
        print(f"🛡️ Request properly blocked: {e}")

if __name__ == "__main__":
    print("🔄 LangChain Lifecycle Hooks Example")
    print("Shows request auditing, security, and monitoring\n")
    
    demo_lifecycle_hooks()
    
    print("\n✅ Lifecycle hooks demo completed!")
    print("🔄 Key concepts demonstrated:")
    print("   - @before_model for pre-processing")
    print("   - @after_model for post-processing") 
    print("   - Runtime access in middleware")
    print("   - Security validation and auditing")
    print("   - Request metrics and monitoring")