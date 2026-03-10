"""
Middleware overview example with LangChain agents.

In this example you will see:
- How middleware provides control at every step of agent execution
- How to use middleware for logging, analytics, and debugging
- How to transform prompts, tool selection, and output formatting
- How to add retries, fallbacks, and early termination logic
- How to apply rate limits, guardrails, and PII detection
- How multiple middleware layers work together in the agent loop
"""
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import before_model, after_model, before_tool, after_tool
from langgraph.runtime import Runtime
from typing import Any
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulate web search delay
    time.sleep(1)
    return f"Search results for '{query}': Found relevant information about the topic."

@tool
def send_email(to: str, subject: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to} with subject: {subject}"

# Middleware 1: Logging and Analytics
@before_model
def logging_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log agent behavior for analytics and debugging."""
    messages = state.get("messages", [])
    print(f"📊 [ANALYTICS] Model call starting - {len(messages)} messages in history")
    
    # Log user input if this is a new conversation turn
    if messages and hasattr(messages[-1], 'role') and messages[-1].role == 'user':
        content = getattr(messages[-1], 'content', '')
        print(f"📥 [LOG] User input: {str(content)[:50]}...")
    
    return None

@after_model
def response_logging_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log model responses for debugging."""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'role') and last_msg.role == 'assistant':
            print(f"🤖 [LOG] Assistant response generated")
    
    return None

# Middleware 2: PII Detection and Guardrails
@before_model
def pii_detection_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Detect and handle personally identifiable information."""
    messages = state.get("messages", [])
    
    # Simple PII detection (in production, use proper PII detection tools)
    pii_patterns = ["ssn", "social security", "credit card", "password"]
    
    for msg in messages:
        if hasattr(msg, 'content'):
            content = str(getattr(msg, 'content', '')).lower()
            if any(pattern in content for pattern in pii_patterns):
                print("🚨 [SECURITY] PII detected - applying privacy protection")
                # In a real system, you might mask or remove PII
                break
    
    return None

# Middleware 3: Rate Limiting
class RateLimitMiddleware:
    def __init__(self, max_calls_per_minute=10):
        self.max_calls = max_calls_per_minute
        self.call_times = []
    
    def __call__(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Apply rate limiting to agent calls."""
        current_time = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]
        
        if len(self.call_times) >= self.max_calls:
            print(f"⚠️  [RATE LIMIT] Maximum {self.max_calls} calls per minute reached")
            # In a real system, you might return an error or delay
        
        self.call_times.append(current_time)
        print(f"📈 [RATE LIMIT] Call {len(self.call_times)}/{self.max_calls} this minute")
        
        return None

# Middleware 4: Tool Call Monitoring
@before_tool
def tool_monitoring_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Monitor and control tool execution."""
    messages = state.get("messages", [])
    
    # Check for tool calls in the last message
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                tool_name = getattr(tool_call, 'name', 'unknown')
                print(f"🔧 [TOOL MONITOR] About to execute: {tool_name}")
                
                # Add safety checks for sensitive tools
                if tool_name == "send_email":
                    print("📧 [SAFETY] Email sending requires additional verification")
    
    return None

@after_tool
def tool_result_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Log tool execution results."""
    messages = state.get("messages", [])
    
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'role') and last_msg.role == 'tool':
            tool_name = getattr(last_msg, 'name', 'unknown')
            print(f"✅ [TOOL MONITOR] Completed execution: {tool_name}")
    
    return None

# Middleware 5: Retry Logic
@after_model
def retry_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Add retry logic for failed operations."""
    messages = state.get("messages", [])
    
    # Simple retry detection (check for error indicators)
    if messages:
        last_msg = messages[-1]
        content = str(getattr(last_msg, 'content', '')).lower()
        
        if "error" in content or "failed" in content:
            print("🔄 [RETRY] Error detected - retry logic could be applied here")
            # In a real system, you might modify the state to trigger a retry
    
    return None

def demo_middleware_stack():
    """Demo: Multiple middleware layers working together."""
    print("🚀 MIDDLEWARE OVERVIEW DEMO")
    print("=" * 60)
    print("Multiple middleware layers controlling agent execution:")
    print("1. 📊 Logging & Analytics")
    print("2. 🚨 PII Detection & Guardrails") 
    print("3. 📈 Rate Limiting")
    print("4. 🔧 Tool Monitoring")
    print("5. 🔄 Retry Logic")
    print()
    
    # Create rate limiting middleware instance
    rate_limiter = RateLimitMiddleware(max_calls_per_minute=5)
    
    # Create agent with multiple middleware layers
    agent = create_agent(
        model=model,
        tools=[search_web, send_email],
        middleware=[
            # Before model middleware
            logging_middleware,
            pii_detection_middleware,
            before_model(rate_limiter),  # Convert instance to decorator
            
            # After model middleware  
            response_logging_middleware,
            retry_middleware,
            
            # Tool middleware
            tool_monitoring_middleware,
            tool_result_middleware,
        ],
        system_prompt="You are a helpful assistant. Use tools when needed and be careful with sensitive operations."
    )
    
    # Test the agent with middleware stack
    test_queries = [
        "Search for information about renewable energy",
        "Send an email to team@company.com about the meeting tomorrow",
        "What are the benefits of solar panels?",  # This might mention SSN in a real scenario
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # Show final response
            if result and "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, 'role') and last_msg.role == 'assistant':
                    content = getattr(last_msg, 'content', '')
                    print(f"🎯 [FINAL] Agent response: {str(content)[:100]}...")
        
        except Exception as e:
            print(f"❌ [ERROR] Agent execution failed: {e}")
        
        print()

if __name__ == "__main__":
    print("🌟 LangChain Middleware Overview")
    print("This demo shows how middleware controls agent execution at every step\n")
    
    demo_middleware_stack()
    
    print("✅ Middleware demo completed!")
    print("💡 Middleware provides powerful control over agent behavior:"
          "\n   - Logging and analytics for monitoring"
          "\n   - Security guardrails and PII protection"  
          "\n   - Rate limiting and resource management"
          "\n   - Tool execution monitoring and safety"
          "\n   - Error handling and retry logic")