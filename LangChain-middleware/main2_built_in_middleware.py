"""
Built-in middleware examples with LangChain agents.

In this example you will see:
- How to use SummarizationMiddleware for conversation history management
- How to use HumanInTheLoopMiddleware for approval workflows
- How to use ModelCallLimitMiddleware to prevent excessive API calls
- How to use ToolCallLimitMiddleware to control tool usage
- How to use ModelFallbackMiddleware for resilient systems
- How to use PIIMiddleware for data protection
- How to use TodoListMiddleware for complex task planning
- How to use LLMToolSelectorMiddleware for intelligent tool selection
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware
)
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
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
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return f"Contents of {path}: Sample file content here..."

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} characters to {path}"

@tool
def run_tests() -> str:
    """Run unit tests."""
    return "All tests passed successfully!"

@tool
def database_query(query: str) -> str:
    """Execute a database query."""
    return f"Database query '{query}' returned 5 rows"

@tool
def weather_lookup(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 22°C"

def demo_summarization_middleware():
    """Demo: SummarizationMiddleware for conversation history management."""
    print("🚀 DEMO 1: Summarization Middleware")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web, weather_lookup],
        middleware=[
            SummarizationMiddleware(
                model=model,  # Use same model for summarization
                trigger={"messages": 5},  # Trigger when 5+ messages
                keep={"messages": 3},     # Keep last 3 messages
            ),
        ],
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful assistant. Keep responses concise."
    )
    
    config = {"configurable": {"thread_id": "summarization_demo"}}
    
    # Simulate a long conversation
    queries = [
        "What's the weather in Paris?",
        "Tell me about renewable energy",
        "Search for information about AI trends",
        "What's the capital of France?",
        "Search for latest tech news",
        "How does summarization work?",  # This should trigger summarization
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config
        )
        
        print(f"📊 Messages in history: {len(result['messages'])}")
        if i >= 5:
            print("🔄 Summarization should have triggered to keep history manageable")

def demo_model_call_limit():
    """Demo: ModelCallLimitMiddleware to prevent excessive API calls."""
    print("\n🚀 DEMO 2: Model Call Limit Middleware")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web],
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=3,      # Max 3 calls per thread
                run_limit=2,         # Max 2 calls per run  
                exit_behavior="end"  # End execution when limit reached
            ),
        ],
        system_prompt="You are a helpful assistant."
    )
    
    config = {"configurable": {"thread_id": "limit_demo"}}
    
    try:
        # This should work (within limit)
        result1 = agent.invoke(
            {"messages": [{"role": "user", "content": "Search for AI news"}]},
            config
        )
        print("✅ First call succeeded")
        
        # This should work (within limit)  
        result2 = agent.invoke(
            {"messages": [{"role": "user", "content": "Search for tech trends"}]},
            config
        )
        print("✅ Second call succeeded")
        
        # This should hit the limit
        result3 = agent.invoke(
            {"messages": [{"role": "user", "content": "Search for startups"}]},
            config
        )
        print("✅ Third call succeeded")
        
    except Exception as e:
        print(f"🚫 Limit reached: {e}")

def demo_tool_call_limit():
    """Demo: ToolCallLimitMiddleware to control tool usage."""
    print("\n🚀 DEMO 3: Tool Call Limit Middleware")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[search_web, database_query],
        middleware=[
            # Global tool limit
            ToolCallLimitMiddleware(thread_limit=4, run_limit=2),
            # Specific tool limit
            ToolCallLimitMiddleware(
                tool_name="database_query",
                thread_limit=1,
                run_limit=1
            ),
        ],
        system_prompt="You are a helpful assistant. Use tools when appropriate."
    )
    
    config = {"configurable": {"thread_id": "tool_limit_demo"}}
    
    # Test tool usage within limits
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search for information and query the database"}]},
        config
    )
    
    print("🔧 Tool call limits applied - excessive tool usage prevented")

def demo_pii_middleware():
    """Demo: PIIMiddleware for data protection."""
    print("\n🚀 DEMO 4: PII Detection Middleware")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[send_email],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        ],
        system_prompt="You are a helpful assistant. Handle sensitive data carefully."
    )
    
    # Test with PII data
    result = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Send email to john.doe@company.com about my credit card 4532-1234-5678-9012"
        }]
    })
    
    print("🔒 PII detection middleware applied - sensitive data protected")

def demo_todo_list_middleware():
    """Demo: TodoListMiddleware for complex task planning."""
    print("\n🚀 DEMO 5: Todo List Middleware")
    print("=" * 60)
    
    agent = create_agent(
        model=model,
        tools=[read_file, write_file, run_tests],
        middleware=[
            TodoListMiddleware()  # Enables task planning and progress tracking
        ],
        system_prompt="You are a helpful assistant that can plan and execute multi-step tasks."
    )
    
    # Complex multi-step task
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "Update the README file to include a Quickstart section, "
                "then save it, and finally run the unit tests to confirm everything works."
            )
        }]
    })
    
    print("📝 Todo list middleware enabled task planning and progress tracking")

def demo_llm_tool_selector():
    """Demo: LLMToolSelectorMiddleware for intelligent tool selection."""
    print("\n🚀 DEMO 6: LLM Tool Selector Middleware")
    print("=" * 60)
    
    # Agent with many tools - selector will choose relevant ones
    agent = create_agent(
        model=model,
        tools=[search_web, send_email, read_file, write_file, database_query, weather_lookup, run_tests],
        middleware=[
            LLMToolSelectorMiddleware(
                model=model,             # LLM for tool selection
                max_tools=3,             # Select max 3 relevant tools
                always_include=["search_web"]  # Always include search capability
            ),
        ],
        system_prompt="You are a helpful assistant with access to various tools."
    )
    
    # Query that should select weather and search tools
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "What's the weather like in Tokyo and can you search for travel tips?"
        }]
    })
    
    print("🎯 Tool selector middleware intelligently chose relevant tools")

if __name__ == "__main__":
    print("🌟 LangChain Built-in Middleware Examples")
    print("This demo shows production-ready middleware for common use cases\n")
    
    # Run all built-in middleware demos
    demo_summarization_middleware()
    demo_model_call_limit() 
    demo_tool_call_limit()
    demo_pii_middleware()
    demo_todo_list_middleware()
    demo_llm_tool_selector()
    
    print("\n✅ All built-in middleware demos completed!")
    print("💡 Built-in middleware provides production-ready solutions for:")
    print("   - 📚 Conversation history management")
    print("   - 🚫 API call and cost control")
    print("   - 🔒 Data privacy and compliance")
    print("   - 📝 Complex task planning")
    print("   - 🎯 Intelligent tool selection")