"""
Lifecycle hooks for intercepting agent execution flow.

Shows how to:
- Use before_agent and after_agent hooks
- Implement before_model and after_model hooks  
- Add logging, monitoring, and instrumentation
- Modify agent behavior at different lifecycle stages
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import (
    before_agent,
    after_agent, 
    before_model,
    after_model
)
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
class SessionContext:
    user_id: str
    session_id: str
    debug_mode: bool

@tool
def calculate_result(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Safe evaluation of basic math expressions
        result = eval(expression.replace("^", "**"))
        return f"Result: {expression} = {result}"
    except:
        return f"Error: Could not calculate '{expression}'"

@tool
def lookup_info(topic: str) -> str:
    """Look up information about a topic."""
    return f"Information about {topic}: This is a comprehensive overview of the topic."

# Global metrics storage (in production, use proper monitoring)
metrics = {
    "requests": 0,
    "model_calls": 0,
    "tool_calls": 0,
    "errors": 0,
    "total_time": 0
}

# 1. Agent lifecycle hooks
@before_agent
def log_agent_start(state, runtime):
    """Log when agent processing starts."""
    ctx = runtime.context
    timestamp = time.time()
    
    print(f"🚀 AGENT START: User {ctx.user_id}, Session {ctx.session_id}")
    print(f"⏱️  Timestamp: {timestamp}")
    
    # Store start time in state for duration calculation
    return {"agent_start_time": timestamp}

@after_agent
def log_agent_complete(state, runtime):
    """Log when agent processing completes."""
    ctx = runtime.context
    start_time = state.get("agent_start_time", time.time())
    duration = time.time() - start_time
    
    print(f"✅ AGENT COMPLETE: Duration {duration:.3f}s")
    
    # Update global metrics
    metrics["requests"] += 1
    metrics["total_time"] += duration
    
    if ctx.debug_mode:
        print(f"🔧 DEBUG: Final state keys: {list(state.keys())}")
    
    return None

# 2. Model lifecycle hooks  
@before_model
def log_model_call(state, runtime):
    """Log before each model call."""
    ctx = runtime.context
    message_count = len(state.get("messages", []))
    
    print(f"🧠 MODEL CALL: {message_count} messages for user {ctx.user_id}")
    
    # Store model call start time
    metrics["model_calls"] += 1
    return {"model_call_time": time.time()}

@after_model  
def log_model_response(state, runtime):
    """Log after each model response."""
    start_time = state.get("model_call_time", time.time())
    duration = time.time() - start_time
    
    print(f"🧠 MODEL RESPONSE: Completed in {duration:.3f}s")
    
    return None

# 3. Advanced hooks with state modification
@before_agent
def security_validation(state, runtime):
    """Validate security before processing requests."""
    ctx = runtime.context
    
    # Check if user is authenticated
    is_authenticated = state.get("authenticated", False)
    
    if not is_authenticated:
        print(f"🔒 SECURITY: User {ctx.user_id} not authenticated")
        # Could raise exception here to block processing
        return {"security_warning": True}
    else:
        print(f"🔒 SECURITY: User {ctx.user_id} authenticated")
        return None

@before_model
def context_injection(state, runtime):
    """Inject context before model calls."""
    ctx = runtime.context
    
    if ctx.debug_mode:
        print(f"🔧 CONTEXT: Debug mode active for user {ctx.user_id}")
        # In a real implementation, you might modify messages here
        return {"debug_context_injected": True}
    
    return None

@after_model
def response_monitoring(state, runtime):
    """Monitor model responses for quality."""
    ctx = runtime.context
    
    # Get the last assistant message
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            content = last_msg.get("content", "")
            
            if len(content) < 10:
                print(f"⚠️  MONITOR: Short response detected ({len(content)} chars)")
            elif len(content) > 500:
                print(f"⚠️  MONITOR: Long response detected ({len(content)} chars)")
            else:
                print(f"✅ MONITOR: Response length appropriate ({len(content)} chars)")
    
    return None

# 4. Error handling hooks
@before_agent
def error_tracking_start(state, runtime):
    """Set up error tracking for the request."""
    return {
        "error_count": 0,
        "error_log": []
    }

@after_agent
def error_tracking_end(state, runtime):
    """Report on any errors that occurred."""
    error_count = state.get("error_count", 0)
    error_log = state.get("error_log", [])
    
    if error_count > 0:
        print(f"❌ ERRORS: {error_count} errors occurred")
        for error in error_log:
            print(f"   - {error}")
        metrics["errors"] += error_count
    else:
        print(f"✅ ERRORS: No errors detected")
    
    return None

def demo_basic_lifecycle_hooks():
    """Demo basic agent and model lifecycle hooks."""
    print("🔄 Basic Lifecycle Hooks")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[calculate_result, lookup_info],
        middleware=[
            log_agent_start,
            log_agent_complete,
            log_model_call,
            log_model_response
        ],
        context_schema=SessionContext,
        system_prompt="You are a helpful assistant with lifecycle logging."
    )
    
    context = SessionContext(user_id="user123", session_id="sess001", debug_mode=False)
    
    print("\n--- Single calculation request ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Calculate 15 * 7 + 23"}]},
        context=context
    )
    
    print("\n--- Information lookup request ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Look up information about quantum computing"}]},
        context=context
    )

def demo_advanced_lifecycle_hooks():
    """Demo advanced hooks with security and monitoring."""
    print("\n🛡️ Advanced Lifecycle Hooks")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[lookup_info],
        middleware=[
            security_validation,
            context_injection,
            response_monitoring,
            error_tracking_start,
            error_tracking_end
        ],
        context_schema=SessionContext,
        system_prompt="You are a secure assistant with advanced monitoring."
    )
    
    # Test with unauthenticated user
    print("\n--- Unauthenticated user request ---")
    unauth_context = SessionContext(user_id="guest", session_id="sess002", debug_mode=True)
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Tell me about artificial intelligence"}],
            "authenticated": False
        },
        context=unauth_context
    )
    
    # Test with authenticated user
    print("\n--- Authenticated user request ---")
    auth_context = SessionContext(user_id="user456", session_id="sess003", debug_mode=False)
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Provide detailed information about machine learning"}],
            "authenticated": True
        },
        context=auth_context
    )

def demo_performance_monitoring():
    """Demo performance monitoring through lifecycle hooks."""
    print("\n📊 Performance Monitoring Hooks")
    print("=" * 50)
    
    @before_agent
    def performance_start(state, runtime):
        """Start performance tracking."""
        return {
            "perf_start": time.time(),
            "perf_checkpoints": []
        }
    
    @before_model
    def performance_checkpoint(state, runtime):
        """Add performance checkpoint before model call."""
        checkpoints = state.get("perf_checkpoints", [])
        checkpoints.append(("model_call", time.time()))
        return {"perf_checkpoints": checkpoints}
    
    @after_model
    def performance_model_complete(state, runtime):
        """Track model call completion."""
        checkpoints = state.get("perf_checkpoints", [])
        checkpoints.append(("model_complete", time.time()))
        return {"perf_checkpoints": checkpoints}
    
    @after_agent
    def performance_report(state, runtime):
        """Generate performance report."""
        start_time = state.get("perf_start", time.time())
        checkpoints = state.get("perf_checkpoints", [])
        total_duration = time.time() - start_time
        
        print(f"📊 PERFORMANCE REPORT:")
        print(f"   Total duration: {total_duration:.3f}s")
        
        for i, (event, timestamp) in enumerate(checkpoints):
            if i == 0:
                duration = timestamp - start_time
            else:
                duration = timestamp - checkpoints[i-1][1]
            print(f"   {event}: {duration:.3f}s")
        
        return None
    
    agent = create_agent(
        model=model,
        tools=[calculate_result],
        middleware=[
            performance_start,
            performance_checkpoint,
            performance_model_complete,
            performance_report
        ],
        context_schema=SessionContext,
        system_prompt="You are an assistant with performance monitoring."
    )
    
    context = SessionContext(user_id="perf_user", session_id="perf_sess", debug_mode=False)
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "Calculate the result of 45 * 23 - 156 + 789"}]},
        context=context
    )

def demo_conditional_hooks():
    """Demo hooks that conditionally modify behavior."""
    print("\n🔀 Conditional Lifecycle Hooks")
    print("=" * 50)
    
    @before_agent
    def conditional_processing(state, runtime):
        """Conditionally modify processing based on context."""
        ctx = runtime.context
        
        if ctx.debug_mode:
            print(f"🔧 DEBUG MODE: Enhanced logging enabled")
            return {"debug_enhanced": True}
        
        if ctx.user_id.startswith("premium_"):
            print(f"⭐ PREMIUM USER: Enhanced features enabled")
            return {"premium_features": True}
        
        return None
    
    @before_model
    def conditional_model_prep(state, runtime):
        """Conditionally prepare model based on flags."""
        if state.get("premium_features"):
            print(f"⭐ PREMIUM: Using enhanced model configuration")
        
        if state.get("debug_enhanced"):
            print(f"🔧 DEBUG: Detailed model call logging active")
        
        return None
    
    agent = create_agent(
        model=model,
        tools=[lookup_info],
        middleware=[
            conditional_processing,
            conditional_model_prep
        ],
        context_schema=SessionContext,
        system_prompt="You are an assistant with conditional features."
    )
    
    # Test regular user
    print("\n--- Regular user ---")
    regular_context = SessionContext(user_id="regular_user", session_id="reg_sess", debug_mode=False)
    agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me about renewable energy"}]},
        context=regular_context
    )
    
    # Test premium user
    print("\n--- Premium user ---")
    premium_context = SessionContext(user_id="premium_user123", session_id="prem_sess", debug_mode=False)
    agent.invoke(
        {"messages": [{"role": "user", "content": "Provide comprehensive analysis of renewable energy"}]},
        context=premium_context
    )
    
    # Test debug user
    print("\n--- Debug mode user ---")
    debug_context = SessionContext(user_id="debug_user", session_id="debug_sess", debug_mode=True)
    agent.invoke(
        {"messages": [{"role": "user", "content": "Simple query about solar energy"}]},
        context=debug_context
    )

def show_metrics():
    """Display accumulated metrics."""
    print("\n📈 Session Metrics Summary")
    print("=" * 50)
    
    avg_time = metrics["total_time"] / max(metrics["requests"], 1)
    
    print(f"Total requests: {metrics['requests']}")
    print(f"Total model calls: {metrics['model_calls']}")
    print(f"Total errors: {metrics['errors']}")
    print(f"Total time: {metrics['total_time']:.3f}s")
    print(f"Average request time: {avg_time:.3f}s")

if __name__ == "__main__":
    print("🔄 LangChain Lifecycle Hooks Example")
    print("Shows how to intercept and monitor agent execution flow\n")
    
    demo_basic_lifecycle_hooks()
    demo_advanced_lifecycle_hooks()
    demo_performance_monitoring()
    demo_conditional_hooks()
    show_metrics()
    
    print("\n✅ Lifecycle hooks demo completed!")
    print("🔄 Key concepts demonstrated:")
    print("   - before_agent and after_agent hooks")
    print("   - before_model and after_model hooks")
    print("   - Security validation and monitoring")
    print("   - Performance tracking and reporting")
    print("   - Conditional behavior modification")
    print("   - Error tracking and metrics collection")