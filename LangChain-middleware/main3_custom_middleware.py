"""
Custom middleware examples with LangChain agents.

In this example you will see:
- How to build custom middleware by implementing hooks at specific execution points
- How to create before_model middleware for input processing and validation
- How to create after_model middleware for output processing and transformation
- How to create before_tool and after_tool middleware for tool execution control
- How to build middleware classes with configuration and state management
- How to combine multiple custom middleware for complex agent behavior
"""
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import before_model, after_model, before_tool, after_tool
from langchain.messages import RemoveMessage, AIMessage
from langgraph.runtime import Runtime
from typing import Any, Dict, List
import time
import re
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    time.sleep(0.5)  # Simulate database latency
    return f"Database search for '{query}': Found 3 matching records"

@tool
def send_notification(message: str, priority: str = "normal") -> str:
    """Send a notification message."""
    return f"Notification sent: {message} (Priority: {priority})"

@tool
def process_payment(amount: float, currency: str = "USD") -> str:
    """Process a payment transaction."""
    return f"Payment processed: {amount} {currency}"

# Custom Middleware 1: Input Validation and Preprocessing
@before_model
def input_validation_middleware(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """Validate and preprocess user input before model processing."""
    messages = state.get("messages", [])
    
    if not messages:
        return None
    
    last_message = messages[-1]
    if hasattr(last_message, 'role') and last_message.role == 'user':
        content = str(getattr(last_message, 'content', ''))
        
        # Input validation
        if len(content) > 1000:
            print("⚠️  [INPUT VALIDATION] Message too long - truncating")
            # In a real system, you might truncate or reject
        
        # Content filtering
        spam_keywords = ['spam', 'advertisement', 'buy now']
        if any(keyword in content.lower() for keyword in spam_keywords):
            print("🚫 [CONTENT FILTER] Potential spam detected")
        
        # Input preprocessing - normalize text
        normalized_content = re.sub(r'\s+', ' ', content.strip())
        print(f"🔧 [PREPROCESSING] Input normalized: {len(content)} -> {len(normalized_content)} chars")
    
    return None

# Custom Middleware 2: Response Enhancement and Post-processing
@after_model
def response_enhancement_middleware(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """Enhance and post-process model responses."""
    messages = state.get("messages", [])
    
    if not messages:
        return None
    
    last_message = messages[-1]
    if hasattr(last_message, 'role') and last_message.role == 'assistant':
        content = getattr(last_message, 'content', '')
        
        # Response quality checks
        if isinstance(content, str) and len(content) < 10:
            print("📏 [QUALITY CHECK] Response seems too short")
        
        # Add helpful formatting
        if isinstance(content, str) and '?' in content:
            print("❓ [ENHANCEMENT] Response contains questions - good for engagement")
        
        print(f"✨ [ENHANCEMENT] Response processed and enhanced")
    
    return None

# Custom Middleware 3: Tool Execution Monitoring
@before_tool
def tool_security_middleware(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """Apply security checks before tool execution."""
    messages = state.get("messages", [])
    
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = getattr(tool_call, 'name', '')
                args = getattr(tool_call, 'args', {})
                
                print(f"🛡️  [SECURITY] Validating tool call: {tool_name}")
                
                # Security checks based on tool type
                if tool_name == "process_payment":
                    amount = args.get('amount', 0)
                    if amount > 1000:
                        print("💰 [SECURITY] High-value payment requires additional approval")
                
                elif tool_name == "send_notification":
                    message = args.get('message', '')
                    if len(message) > 200:
                        print("📢 [SECURITY] Long notification message detected")
    
    return None

@after_tool
def tool_performance_middleware(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """Monitor tool performance and log execution metrics."""
    messages = state.get("messages", [])
    
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'role') and last_message.role == 'tool':
            tool_name = getattr(last_message, 'name', 'unknown')
            content = getattr(last_message, 'content', '')
            
            # Performance monitoring
            execution_time = 0.5  # In a real system, you'd measure actual time
            print(f"⏱️  [PERFORMANCE] Tool '{tool_name}' executed in {execution_time}s")
            
            # Result validation
            if isinstance(content, str) and 'error' in content.lower():
                print(f"❌ [MONITORING] Tool '{tool_name}' returned error")
            else:
                print(f"✅ [MONITORING] Tool '{tool_name}' completed successfully")
    
    return None

# Custom Middleware Class: Conversation Analytics
class ConversationAnalyticsMiddleware:
    """Track conversation metrics and analytics."""
    
    def __init__(self):
        self.metrics = {
            'total_messages': 0,
            'tool_calls': 0,
            'conversation_turns': 0,
            'avg_response_length': 0
        }
        self.response_lengths = []
    
    @before_model
    def track_conversation_start(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Track when a conversation turn begins."""
        messages = state.get("messages", [])
        
        # Count user messages to track conversation turns
        user_messages = [m for m in messages if hasattr(m, 'role') and m.role == 'user']
        self.metrics['conversation_turns'] = len(user_messages)
        
        print(f"📊 [ANALYTICS] Conversation turn {self.metrics['conversation_turns']} starting")
        return None
    
    @after_model  
    def track_response_metrics(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Track response metrics and analytics."""
        messages = state.get("messages", [])
        self.metrics['total_messages'] = len(messages)
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'role') and last_message.role == 'assistant':
                content = getattr(last_message, 'content', '')
                if isinstance(content, str):
                    self.response_lengths.append(len(content))
                    self.metrics['avg_response_length'] = sum(self.response_lengths) / len(self.response_lengths)
        
        print(f"📈 [ANALYTICS] Current metrics: {json.dumps(self.metrics, indent=2)}")
        return None
    
    @after_tool
    def track_tool_usage(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Track tool usage statistics."""
        self.metrics['tool_calls'] += 1
        print(f"🔧 [ANALYTICS] Tool call #{self.metrics['tool_calls']} completed")
        return None

# Custom Middleware: Response Caching
class ResponseCachingMiddleware:
    """Cache responses for identical queries to improve performance."""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    @before_model
    def check_cache(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Check if we have a cached response for this query."""
        messages = state.get("messages", [])
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'role') and last_message.role == 'user':
                query = str(getattr(last_message, 'content', ''))
                query_hash = hash(query)
                
                if query_hash in self.cache:
                    cached_response = self.cache[query_hash]
                    self.cache_hits += 1
                    print(f"💾 [CACHE] Cache hit! Using cached response ({self.cache_hits} hits)")
                    
                    # Return cached response by adding it to messages
                    return {
                        "messages": [
                            AIMessage(content=cached_response, role='assistant')
                        ]
                    }
                else:
                    self.cache_misses += 1
                    print(f"🔍 [CACHE] Cache miss. Will cache response ({self.cache_misses} misses)")
        
        return None
    
    @after_model
    def cache_response(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Cache the model response for future use."""
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            # Get user query and assistant response
            user_msg = messages[-2] if hasattr(messages[-2], 'role') and messages[-2].role == 'user' else None
            assistant_msg = messages[-1] if hasattr(messages[-1], 'role') and messages[-1].role == 'assistant' else None
            
            if user_msg and assistant_msg:
                query = str(getattr(user_msg, 'content', ''))
                response = str(getattr(assistant_msg, 'content', ''))
                query_hash = hash(query)
                
                # Cache management - remove oldest if at capacity
                if len(self.cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    print("🗑️  [CACHE] Removed oldest cached entry")
                
                self.cache[query_hash] = response
                print(f"💾 [CACHE] Cached response ({len(self.cache)} entries)")
        
        return None

def demo_custom_middleware():
    """Demo: Multiple custom middleware working together."""
    print("🚀 CUSTOM MIDDLEWARE DEMO")
    print("=" * 60)
    
    # Create analytics and caching middleware instances
    analytics = ConversationAnalyticsMiddleware()
    cache = ResponseCachingMiddleware(max_cache_size=5)
    
    # Create agent with multiple custom middleware
    agent = create_agent(
        model=model,
        tools=[search_database, send_notification, process_payment],
        middleware=[
            # Input processing
            input_validation_middleware,
            
            # Analytics middleware (using instance methods)
            analytics.track_conversation_start,
            analytics.track_response_metrics,
            analytics.track_tool_usage,
            
            # Caching middleware
            cache.check_cache,
            cache.cache_response,
            
            # Response processing
            response_enhancement_middleware,
            
            # Tool monitoring
            tool_security_middleware,
            tool_performance_middleware,
        ],
        system_prompt="You are a helpful assistant with advanced middleware capabilities."
    )
    
    # Test queries to demonstrate middleware functionality
    test_queries = [
        "Search the database for user accounts",
        "Send a notification about system maintenance",
        "Search the database for user accounts",  # This should hit cache
        "Process a payment of 500 USD",
        "What's the status of the system?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Show final response
        if result and "messages" in result:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'role') and last_msg.role == 'assistant':
                content = getattr(last_msg, 'content', '')
                print(f"🎯 [FINAL] Response: {str(content)[:80]}...")
    
    # Show final analytics
    print(f"\n📊 [FINAL ANALYTICS] {json.dumps(analytics.metrics, indent=2)}")
    print(f"💾 [CACHE STATS] Hits: {cache.cache_hits}, Misses: {cache.cache_misses}")

if __name__ == "__main__":
    print("🌟 LangChain Custom Middleware Examples")
    print("This demo shows how to build custom middleware for specialized agent behavior\n")
    
    demo_custom_middleware()
    
    print("\n✅ Custom middleware demo completed!")
    print("💡 Custom middleware enables specialized behaviors:")
    print("   - 🔧 Input validation and preprocessing")
    print("   - ✨ Response enhancement and post-processing")
    print("   - 🛡️  Tool security and performance monitoring")
    print("   - 📊 Conversation analytics and metrics")
    print("   - 💾 Response caching and optimization")