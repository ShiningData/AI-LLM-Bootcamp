"""
Dynamic response format selection based on conversation state and user context.

Shows how to:
- Define structured output schemas with Pydantic
- Select response format based on conversation length
- Adapt output format based on user preferences
- Use different schemas for different use cases
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing import Callable, List
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=400
)

@dataclass
class UserContext:
    user_id: str
    role: str
    output_preference: str  # simple, detailed, structured

# Define different response format schemas
class SimpleResponse(BaseModel):
    """Simple response for quick answers."""
    answer: str = Field(description="A brief, direct answer")

class DetailedResponse(BaseModel):
    """Detailed response with reasoning."""
    answer: str = Field(description="A comprehensive answer")
    reasoning: str = Field(description="Explanation of the reasoning process")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class AnalysisResponse(BaseModel):
    """Structured analysis response."""
    summary: str = Field(description="Brief summary of findings")
    key_points: List[str] = Field(description="List of key insights")
    recommendations: List[str] = Field(description="Actionable recommendations")
    confidence: float = Field(description="Overall confidence in analysis")

class CustomerSupportTicket(BaseModel):
    """Structured customer support ticket."""
    category: str = Field(description="Issue category: 'billing', 'technical', 'account', or 'product'")
    priority: str = Field(description="Urgency level: 'low', 'medium', 'high', or 'critical'")
    summary: str = Field(description="One-sentence summary of the issue")
    sentiment: str = Field(description="Customer sentiment: 'frustrated', 'neutral', or 'satisfied'")
    next_steps: List[str] = Field(description="Recommended actions to resolve the issue")

class TaskPlan(BaseModel):
    """Structured task planning response."""
    goal: str = Field(description="Main objective")
    tasks: List[str] = Field(description="List of specific tasks to complete")
    estimated_time: str = Field(description="Estimated completion time")
    priority_order: List[int] = Field(description="Task execution order by priority")

@tool
def analyze_issue(description: str) -> str:
    """Analyze a customer support issue."""
    return f"Analyzed issue: {description}. Identified patterns and potential solutions."

@tool
def create_plan(objective: str) -> str:
    """Create a task plan for an objective."""
    return f"Created plan for: {objective}. Outlined steps and timeline."

# 1. Conversation state-based response format
@wrap_model_call
def conversation_state_response_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select response format based on conversation progress."""
    
    message_count = len(request.messages)
    
    if message_count < 3:
        # Early conversation - use simple format
        response_format = SimpleResponse
        format_type = "simple (early conversation)"
    elif message_count < 8:
        # Established conversation - use detailed format
        response_format = DetailedResponse
        format_type = "detailed (established conversation)"
    else:
        # Long conversation - use analysis format
        response_format = AnalysisResponse
        format_type = "analysis (extended conversation)"
    
    print(f"📊 FORMAT: Using {format_type} for {message_count} messages")
    request = request.override(response_format=response_format)
    
    return handler(request)

# 2. User role-based response format
@wrap_model_call
def role_based_response_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select response format based on user role."""
    
    user_role = request.runtime.context.role
    
    if user_role == "customer_support":
        response_format = CustomerSupportTicket
        format_type = "customer support ticket"
    elif user_role == "project_manager":
        response_format = TaskPlan
        format_type = "task plan"
    elif user_role == "analyst":
        response_format = AnalysisResponse
        format_type = "analysis report"
    else:
        response_format = DetailedResponse
        format_type = "detailed (default)"
    
    print(f"👤 ROLE: Using {format_type} format for {user_role} role")
    request = request.override(response_format=response_format)
    
    return handler(request)

# 3. User preference-based response format
@wrap_model_call
def preference_based_response_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select response format based on user preferences."""
    
    output_pref = request.runtime.context.output_preference
    
    if output_pref == "simple":
        response_format = SimpleResponse
        format_type = "simple"
    elif output_pref == "structured":
        response_format = AnalysisResponse
        format_type = "structured analysis"
    else:  # detailed
        response_format = DetailedResponse
        format_type = "detailed with reasoning"
    
    print(f"⚙️ PREF: Using {format_type} format per user preference")
    request = request.override(response_format=response_format)
    
    return handler(request)

# 4. Content-aware response format
@wrap_model_call
def content_aware_response_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select response format based on message content."""
    
    # Get the last user message
    last_message = ""
    for msg in reversed(request.messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_message = msg.get("content", "").lower()
            break
    
    # Select format based on content keywords
    if any(word in last_message for word in ["support", "issue", "problem", "help"]):
        response_format = CustomerSupportTicket
        format_type = "support ticket"
    elif any(word in last_message for word in ["plan", "task", "project", "schedule"]):
        response_format = TaskPlan
        format_type = "task plan"
    elif any(word in last_message for word in ["analyze", "analysis", "insights", "data"]):
        response_format = AnalysisResponse
        format_type = "analysis"
    elif any(word in last_message for word in ["quick", "brief", "short"]):
        response_format = SimpleResponse
        format_type = "simple"
    else:
        response_format = DetailedResponse
        format_type = "detailed"
    
    print(f"🔍 CONTENT: Selected {format_type} format based on content keywords")
    request = request.override(response_format=response_format)
    
    return handler(request)

def demo_conversation_state_formats():
    """Demo response format selection based on conversation state."""
    print("📊 Conversation State-Based Response Formats")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[analyze_issue],
        middleware=[conversation_state_response_format],
        context_schema=UserContext,
        system_prompt="You are an adaptive assistant that provides structured responses."
    )
    
    context = UserContext(user_id="user1", role="user", output_preference="detailed")
    
    # Early conversation
    print("\n--- Early conversation (simple format) ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is machine learning?"}]},
        context=context
    )
    
    # Medium conversation
    print("\n--- Established conversation (detailed format) ---")
    medium_messages = [{"role": "user", "content": f"Question {i}"} for i in range(5)]
    result = agent.invoke(
        {"messages": medium_messages + [{"role": "user", "content": "Explain neural networks in depth"}]},
        context=context
    )
    
    # Long conversation  
    print("\n--- Extended conversation (analysis format) ---")
    long_messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
    result = agent.invoke(
        {"messages": long_messages + [{"role": "user", "content": "Analyze the advantages of deep learning"}]},
        context=context
    )

def demo_role_based_formats():
    """Demo response format selection based on user roles."""
    print("\n👤 Role-Based Response Formats")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[analyze_issue, create_plan],
        middleware=[role_based_response_format],
        context_schema=UserContext,
        system_prompt="You are a role-aware assistant."
    )
    
    # Customer support role
    print("\n--- Customer support role ---")
    support_context = UserContext(user_id="support1", role="customer_support", output_preference="structured")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Customer is having trouble logging into their account and seems frustrated"}]},
        context=support_context
    )
    
    # Project manager role
    print("\n--- Project manager role ---")
    pm_context = UserContext(user_id="pm1", role="project_manager", output_preference="structured")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Help me plan the development of a new mobile app feature"}]},
        context=pm_context
    )
    
    # Analyst role
    print("\n--- Analyst role ---")
    analyst_context = UserContext(user_id="analyst1", role="analyst", output_preference="detailed")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the trends in user engagement this quarter?"}]},
        context=analyst_context
    )

def demo_preference_based_formats():
    """Demo response format selection based on user preferences."""
    print("\n⚙️ Preference-Based Response Formats")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[preference_based_response_format],
        context_schema=UserContext,
        system_prompt="You adapt your response format to user preferences."
    )
    
    # Simple preference
    print("\n--- User prefers simple responses ---")
    simple_context = UserContext(user_id="user1", role="user", output_preference="simple")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the benefits of cloud computing?"}]},
        context=simple_context
    )
    
    # Structured preference
    print("\n--- User prefers structured responses ---")
    structured_context = UserContext(user_id="user2", role="user", output_preference="structured")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the benefits of cloud computing?"}]},
        context=structured_context
    )

def demo_content_aware_formats():
    """Demo response format selection based on message content."""
    print("\n🔍 Content-Aware Response Formats")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[analyze_issue, create_plan],
        middleware=[content_aware_response_format],
        context_schema=UserContext,
        system_prompt="You adapt response format based on the type of question asked."
    )
    
    context = UserContext(user_id="user1", role="user", output_preference="detailed")
    
    # Support-related query
    print("\n--- Support-related query ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I need help with a billing issue that's been frustrating me"}]},
        context=context
    )
    
    # Planning-related query
    print("\n--- Planning-related query ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Help me create a project plan for launching our new product"}]},
        context=context
    )
    
    # Analysis-related query
    print("\n--- Analysis-related query ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Analyze the data trends and provide insights for our marketing strategy"}]},
        context=context
    )

if __name__ == "__main__":
    print("📊 LangChain Dynamic Response Format Example")
    print("Shows how to structure outputs based on context and preferences\n")
    
    demo_conversation_state_formats()
    demo_role_based_formats()
    demo_preference_based_formats()
    demo_content_aware_formats()
    
    print("\n✅ Dynamic response format demo completed!")
    print("📊 Key concepts demonstrated:")
    print("   - Pydantic schema definitions")
    print("   - Conversation state-based format selection")
    print("   - Role-specific output structures")
    print("   - User preference adaptation")
    print("   - Content-aware format matching")
    print("   - request.override(response_format=...) modification")