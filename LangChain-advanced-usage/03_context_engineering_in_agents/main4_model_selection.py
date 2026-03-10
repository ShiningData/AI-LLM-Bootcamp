"""
Dynamic model selection based on conversation state and context.

Shows how to:
- Select models based on conversation length
- Choose models based on user preferences
- Switch models based on task complexity
- Optimize costs with appropriate model selection
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

# Initialize multiple models for different use cases
efficient_model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=200
)

standard_model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

large_model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=1000
)

@dataclass
class TaskContext:
    user_id: str
    task_complexity: str  # simple, medium, complex
    priority: str         # low, high

@tool
def analyze_data(dataset: str) -> str:
    """Analyze a dataset (complex task)."""
    return f"Analysis of {dataset}: Found patterns and insights."

@tool
def simple_lookup(item: str) -> str:
    """Simple information lookup (simple task)."""
    return f"Information about {item}: Basic details found."

@tool
def generate_report(topic: str) -> str:
    """Generate detailed report (medium-complex task)."""
    return f"Generated comprehensive report on {topic}."

# 1. Conversation length-based model selection
@wrap_model_call
def conversation_length_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on conversation length."""
    
    message_count = len(request.messages)
    
    if message_count > 20:
        # Long conversation - use model with larger context window
        selected_model = large_model
        model_type = "large (extended context)"
    elif message_count > 10:
        # Medium conversation - use standard model
        selected_model = standard_model
        model_type = "standard"
    else:
        # Short conversation - use efficient model
        selected_model = efficient_model
        model_type = "efficient"
    
    print(f"🧠 MODEL: Selected {model_type} model for conversation with {message_count} messages")
    request = request.override(model=selected_model)
    
    return handler(request)

# 2. Task complexity-based model selection
@wrap_model_call
def task_complexity_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on task complexity from context."""
    
    task_complexity = request.runtime.context.task_complexity
    priority = request.runtime.context.priority
    
    # Select model based on complexity and priority
    if task_complexity == "complex" or priority == "high":
        selected_model = large_model
        model_type = "large (complex/high-priority)"
    elif task_complexity == "medium":
        selected_model = standard_model  
        model_type = "standard (medium complexity)"
    else:
        selected_model = efficient_model
        model_type = "efficient (simple tasks)"
    
    print(f"🎯 TASK: Using {model_type} model for {task_complexity} complexity, {priority} priority task")
    request = request.override(model=selected_model)
    
    return handler(request)

# 3. Tool-based model selection
@wrap_model_call
def tool_aware_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on tools being used."""
    
    # Check if complex analysis tools are available
    tool_names = [tool.name for tool in request.tools]
    
    has_analysis_tools = any("analyze" in name for name in tool_names)
    has_reporting_tools = any("report" in name for name in tool_names)
    
    if has_analysis_tools:
        # Complex analysis requires larger model
        selected_model = large_model
        model_type = "large (analysis tools)"
    elif has_reporting_tools:
        # Reporting requires medium model
        selected_model = standard_model
        model_type = "standard (reporting tools)"
    else:
        # Simple tools can use efficient model
        selected_model = efficient_model
        model_type = "efficient (simple tools)"
    
    print(f"🛠️ TOOLS: Selected {model_type} based on available tools: {tool_names}")
    request = request.override(model=selected_model)
    
    return handler(request)

# 4. User preference-based model selection
@wrap_model_call
def preference_based_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on user preferences stored in memory."""
    
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    
    # Default to efficient model
    selected_model = efficient_model
    model_type = "efficient (default)"
    
    if store:
        try:
            user_prefs = store.get(("model_preferences",), user_id)
            
            if user_prefs:
                preferred_quality = user_prefs.value.get("quality_preference", "balanced")
                cost_sensitivity = user_prefs.value.get("cost_sensitive", False)
                
                if preferred_quality == "high" and not cost_sensitivity:
                    selected_model = large_model
                    model_type = "large (user prefers quality)"
                elif preferred_quality == "balanced":
                    selected_model = standard_model
                    model_type = "standard (user prefers balance)"
                # else keep efficient model for cost-sensitive users
                
                print(f"👤 USER: Selected {model_type} based on user preferences")
        except:
            print("👤 USER: Using default model (no preferences stored)")
    
    request = request.override(model=selected_model)
    return handler(request)

def demo_conversation_length_selection():
    """Demo model selection based on conversation length."""
    print("📏 Conversation Length-Based Model Selection")
    print("=" * 50)
    
    agent = create_agent(
        model=efficient_model,  # Default model (will be overridden)
        tools=[simple_lookup],
        middleware=[conversation_length_model_selection],
        context_schema=TaskContext,
        system_prompt="You are an adaptive assistant."
    )
    
    context = TaskContext(user_id="user1", task_complexity="simple", priority="low")
    
    # Short conversation
    print("\n--- Short conversation ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "What is Python?"}]},
        context=context
    )
    
    # Medium conversation
    print("\n--- Medium conversation ---")
    medium_messages = [{"role": "user", "content": f"Question {i}"} for i in range(12)]
    agent.invoke(
        {"messages": medium_messages + [{"role": "user", "content": "Explain machine learning"}]},
        context=context
    )
    
    # Long conversation
    print("\n--- Long conversation ---")
    long_messages = [{"role": "user", "content": f"Message {i}"} for i in range(25)]
    agent.invoke(
        {"messages": long_messages + [{"role": "user", "content": "Summarize our discussion"}]},
        context=context
    )

def demo_task_complexity_selection():
    """Demo model selection based on task complexity."""
    print("\n🎯 Task Complexity-Based Model Selection")
    print("=" * 50)
    
    agent = create_agent(
        model=efficient_model,
        tools=[simple_lookup, analyze_data, generate_report],
        middleware=[task_complexity_model_selection],
        context_schema=TaskContext,
        system_prompt="You are a task-aware assistant."
    )
    
    # Simple task
    print("\n--- Simple task ---")
    simple_context = TaskContext(user_id="user1", task_complexity="simple", priority="low")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Look up basic information about AI"}]},
        context=simple_context
    )
    
    # Complex task
    print("\n--- Complex task ---")
    complex_context = TaskContext(user_id="user1", task_complexity="complex", priority="high")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Analyze this dataset and provide insights"}]},
        context=complex_context
    )

def demo_tool_aware_selection():
    """Demo model selection based on available tools."""
    print("\n🛠️ Tool-Aware Model Selection")
    print("=" * 50)
    
    context = TaskContext(user_id="user1", task_complexity="medium", priority="medium")
    
    # Agent with simple tools
    print("\n--- Agent with simple tools ---")
    simple_agent = create_agent(
        model=efficient_model,
        tools=[simple_lookup],
        middleware=[tool_aware_model_selection],
        context_schema=TaskContext,
        system_prompt="Simple assistant."
    )
    simple_agent.invoke(
        {"messages": [{"role": "user", "content": "Help me with basic lookup"}]},
        context=context
    )
    
    # Agent with analysis tools
    print("\n--- Agent with analysis tools ---")
    analysis_agent = create_agent(
        model=efficient_model,
        tools=[analyze_data, generate_report],
        middleware=[tool_aware_model_selection],
        context_schema=TaskContext,
        system_prompt="Analysis assistant."
    )
    analysis_agent.invoke(
        {"messages": [{"role": "user", "content": "Analyze data and create reports"}]},
        context=context
    )

def demo_preference_based_selection():
    """Demo model selection based on user preferences."""
    print("\n👤 Preference-Based Model Selection")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Set up user preferences
    store.put(("model_preferences",), "quality_user", {
        "quality_preference": "high",
        "cost_sensitive": False
    })
    
    store.put(("model_preferences",), "cost_user", {
        "quality_preference": "low",
        "cost_sensitive": True
    })
    
    agent = create_agent(
        model=efficient_model,
        tools=[simple_lookup],
        middleware=[preference_based_model_selection],
        context_schema=TaskContext,
        store=store,
        system_prompt="Preference-aware assistant."
    )
    
    # Quality-focused user
    print("\n--- Quality-focused user ---")
    quality_context = TaskContext(user_id="quality_user", task_complexity="simple", priority="low")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me understand neural networks"}]},
        context=quality_context
    )
    
    # Cost-sensitive user
    print("\n--- Cost-sensitive user ---")
    cost_context = TaskContext(user_id="cost_user", task_complexity="simple", priority="low")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Quick question about Python"}]},
        context=cost_context
    )

if __name__ == "__main__":
    print("🧠 LangChain Dynamic Model Selection Example")
    print("Shows how to choose the right model for different scenarios\n")
    
    demo_conversation_length_selection()
    demo_task_complexity_selection()
    demo_tool_aware_selection()
    demo_preference_based_selection()
    
    print("\n✅ Dynamic model selection demo completed!")
    print("🧠 Key concepts demonstrated:")
    print("   - Conversation length-based selection")
    print("   - Task complexity optimization")
    print("   - Tool-aware model matching")
    print("   - User preference-based selection")
    print("   - Cost vs quality trade-offs")