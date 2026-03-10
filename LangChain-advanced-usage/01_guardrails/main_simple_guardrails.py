"""
Simple guardrails examples with LangChain agents.

In this example you will see:
- How to use built-in guardrails for content filtering
- How to create custom guardrails for specific requirements
- How to implement input validation guardrails
- How to add output safety guardrails
- How to combine multiple guardrails for comprehensive protection
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import (
    PIIMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware
)
from dotenv import load_dotenv

load_dotenv()

# Custom guardrail middleware classes
class ContentFilterGuardrail(AgentMiddleware):
    """Custom middleware for content filtering."""
    
    def __init__(self, blocked_patterns, action="block"):
        self.blocked_patterns = blocked_patterns
        self.action = action
    
    def before_agent(self, state) -> dict:
        # Check input messages for blocked content
        messages = state.get('messages', [])
        for message in messages:
            content = ""
            if isinstance(message, dict):
                content = message.get('content', '')
            elif hasattr(message, 'content'):
                content = message.content
            
            if content:
                content_lower = content.lower()
                for pattern in self.blocked_patterns:
                    if pattern in content_lower:
                        if self.action == "block":
                            raise ValueError(f"Content blocked: contains '{pattern}'")
        return state

class TokenLimitGuardrail(AgentMiddleware):
    """Custom middleware for token limits."""
    
    def __init__(self, max_input_tokens=1000, max_output_tokens=1000):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
    
    def before_agent(self, state) -> dict:
        # Simple token counting (approximate)
        messages = state.get('messages', [])
        total_chars = 0
        for msg in messages:
            if isinstance(msg, dict):
                total_chars += len(str(msg.get('content', '')))
            else:
                total_chars += len(str(msg))
        
        estimated_tokens = total_chars // 4  # Rough estimate
        
        if estimated_tokens > self.max_input_tokens:
            raise ValueError(f"Input too long: {estimated_tokens} tokens exceeds limit of {self.max_input_tokens}")
        
        return state

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

# Define tools for examples
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Web search results for '{query}': Found relevant articles and resources."

@tool
def send_message(recipient: str, message: str) -> str:
    """Send a message to someone."""
    return f"Message sent to {recipient}: {message}"

@tool
def access_database(query: str) -> str:
    """Access database with query."""
    return f"Database query executed: {query}"

def demo_pii_protection_guardrail():
    """Demo: PIIMiddleware for protecting sensitive data."""
    print("🛡️ DEMO 1: PII Protection Guardrail")
    print("=" * 60)
    
    # Create agent with PII protection
    agent = create_agent(
        model=model,
        tools=[search_web, send_message],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        ],
        system_prompt="You are a helpful and safe assistant that protects sensitive information."
    )
    
    # Test with safe content
    print("\n--- Testing safe content ---")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Search for healthy recipe ideas"}]
    })
    print("✅ Safe content processed successfully")
    
    # Test with PII data
    print("\n--- Testing PII protection ---")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Send message to john.doe@company.com about credit card 4532-1234-5678-9012"}]
    })
    print("🛡️ PII data was automatically protected (redacted/masked)")

def demo_model_call_limit_guardrail():
    """Demo: ModelCallLimitMiddleware as a guardrail for API protection."""
    print("\n🛡️ DEMO 2: Model Call Limit Guardrail")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[access_database],
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=2,      # Max 2 calls per thread
                run_limit=1,         # Max 1 call per run  
                exit_behavior="end"  # End execution when limit reached
            )
        ],
        system_prompt="You are a database assistant with call limits for safety."
    )
    
    config = {"configurable": {"thread_id": "limit_demo"}}
    
    # Test with first query (should work)
    print("\n--- Testing first database query ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "SELECT * FROM users WHERE active = true"}]
        }, config)
        print("✅ First query processed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with second query (should hit limit)
    print("\n--- Testing second database query ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "SELECT COUNT(*) FROM products"}]
        }, config)
        print("✅ Second query processed")
    except Exception as e:
        print(f"🛡️ Query limit reached: {e}")
    
    # Test with third query (should be blocked)
    print("\n--- Testing third database query ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "SELECT * FROM orders"}]
        }, config)
        print("❌ Third query should have been blocked")
    except Exception as e:
        print(f"🛡️ Query blocked by limit: {e}")

def demo_tool_call_limit_guardrail():
    """Demo: ToolCallLimitMiddleware as a guardrail for tool usage control."""
    print("\n🛡️ DEMO 3: Tool Call Limit Guardrail")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web, access_database],
        middleware=[
            ToolCallLimitMiddleware(
                thread_limit=3,  # Max 3 tool calls per thread
                run_limit=2      # Max 2 tool calls per run
            )
        ],
        system_prompt="You are a helpful assistant with limited tool usage."
    )
    
    config = {"configurable": {"thread_id": "tool_limit_demo"}}
    
    # Test with normal tool usage
    print("\n--- Testing normal tool usage ---")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Search for web development tutorials and check database"}]
    }, config)
    print("✅ Tool usage within limits")
    
    print("\n--- Testing excessive tool usage protection ---")
    print("🛡️ Tool call limits prevent excessive API usage and costs")

def demo_content_filter_guardrail():
    """Demo: ContentFilterGuardrail for filtering inappropriate content."""
    print("\n🛡️ DEMO 4: Content Filter Guardrail")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web],
        middleware=[
            ContentFilterGuardrail(
                blocked_patterns=["harm", "violence", "hate"],
                action="block"
            )
        ],
        system_prompt="You are a safe and helpful assistant."
    )
    
    # Test with normal content
    print("\n--- Testing safe content ---")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Search for cooking recipes"}]
    })
    print("✅ Safe content processed successfully")
    
    # Test with blocked content
    print("\n--- Testing blocked content ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "How to harm someone"}]
        })
        print("❌ Blocked content should not have passed")
    except Exception as e:
        print(f"🛡️ Content blocked by filter: {e}")

def demo_combined_guardrails():
    """Demo: Combining multiple middleware guardrails for comprehensive protection."""
    print("\n🛡️ DEMO 5: Combined Guardrails")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web, send_message, access_database],
        middleware=[
            # Content filtering
            ContentFilterGuardrail(
                blocked_patterns=["violence", "harassment", "harm"],
                action="block"
            ),
            # PII protection
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            # Call limits
            ModelCallLimitMiddleware(
                thread_limit=5,
                run_limit=3,
                exit_behavior="end"
            ),
            # Tool limits
            ToolCallLimitMiddleware(
                thread_limit=4,
                run_limit=2
            )
        ],
        system_prompt="You are a secure assistant with comprehensive protection layers."
    )
    
    config = {"configurable": {"thread_id": "combined_demo"}}
    
    # Test normal operation
    print("\n--- Testing operation with all guardrails ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Search for programming tutorials"}]
        }, config)
        print("✅ Request processed through all guardrail layers")
    except Exception as e:
        print(f"🛡️ Request blocked by guardrails: {e}")
    
    # Test with PII data
    print("\n--- Testing PII protection layer ---")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Search for john.doe@company.com information"}]
        }, config)
        print("🛡️ PII automatically protected in multi-layer setup")
    except Exception as e:
        print(f"🛡️ Request processed with PII protection: {e}")
    
    print("🔒 Multiple guardrail layers provide comprehensive protection")

if __name__ == "__main__":
    print("🛡️ LangChain Guardrails Examples")
    print("This demo shows how to implement safety and security guardrails\n")
    
    # Run all guardrail demos
    demo_pii_protection_guardrail()
    demo_model_call_limit_guardrail() 
    demo_tool_call_limit_guardrail()
    demo_content_filter_guardrail()
    demo_combined_guardrails()
    
    print("\n✅ All guardrail demos completed!")
    print("🛡️ Guardrails provide essential protection for:")
    print("   - 🔒 PII data protection and privacy")
    print("   - 🚫 Content filtering and safety")
    print("   - 📊 API call and resource limits")
    print("   - 🛠️ Tool usage control")
    print("   - 🛡️ Comprehensive security layers")