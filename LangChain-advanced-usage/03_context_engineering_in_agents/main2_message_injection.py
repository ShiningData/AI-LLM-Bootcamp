"""
Message context injection using wrap_model_call middleware.

Shows how to:
- Inject file context from state
- Add user writing style from store
- Include compliance rules from runtime context
- Modify messages before they reach the model
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
class ComplianceContext:
    user_id: str
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]

@tool
def document_search(query: str) -> str:
    """Search through uploaded documents."""
    return f"Document search results for '{query}': Found relevant content in uploaded files."

# 1. Inject file context from agent state
@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject information about uploaded files into the model request."""
    
    # Get uploaded files from agent state
    uploaded_files = request.state.get("uploaded_files", [])
    
    if uploaded_files:
        # Build file descriptions
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )
        
        # Create context message
        file_context = f"""Files available in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""
        
        # Add context as a user message
        messages = [
            *request.messages,
            {"role": "user", "content": file_context}
        ]
        
        request = request.override(messages=messages)
    
    return handler(request)

# 2. Inject writing style from store
@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject user's writing style preferences from store."""
    
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    
    if store:
        try:
            writing_style = store.get(("writing_style",), user_id)
            
            if writing_style:
                style = writing_style.value
                
                style_context = f"""Your writing style preferences:
- Tone: {style.get('tone', 'professional')}
- Greeting: "{style.get('greeting', 'Hi')}"
- Sign-off: "{style.get('sign_off', 'Best regards')}"
- Example: {style.get('example', 'N/A')}

Match this style in your responses."""
                
                messages = [
                    *request.messages,
                    {"role": "user", "content": style_context}
                ]
                
                request = request.override(messages=messages)
        except:
            pass  # No writing style stored
    
    return handler(request)

# 3. Inject compliance rules from runtime context
@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject compliance constraints based on user's jurisdiction and industry."""
    
    ctx = request.runtime.context
    jurisdiction = ctx.user_jurisdiction
    industry = ctx.industry
    frameworks = ctx.compliance_frameworks
    
    rules = []
    
    # GDPR rules
    if "GDPR" in frameworks:
        rules.append("- Must obtain explicit consent before processing personal data")
        rules.append("- Users have the right to request data deletion")
    
    # HIPAA rules
    if "HIPAA" in frameworks:
        rules.append("- Cannot share patient health information without authorization")
        rules.append("- Must use secure, encrypted communication channels")
    
    # Industry-specific rules
    if industry == "finance":
        rules.append("- Cannot provide financial advice without proper disclaimers")
        rules.append("- All financial data must be handled with extra security")
    elif industry == "healthcare":
        rules.append("- Patient privacy is paramount")
        rules.append("- Medical advice requires proper qualification disclaimers")
    
    if rules:
        compliance_context = f"""Compliance requirements for {jurisdiction} ({industry}):
{chr(10).join(rules)}

Ensure all responses comply with these requirements."""
        
        messages = [
            *request.messages,
            {"role": "user", "content": compliance_context}
        ]
        
        request = request.override(messages=messages)
    
    return handler(request)

def demo_file_context_injection():
    """Demo injecting file context from agent state."""
    print("📁 File Context Injection")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[document_search],
        middleware=[inject_file_context],
        context_schema=ComplianceContext,
        system_prompt="You are a document assistant with access to uploaded files."
    )
    
    # Simulate uploaded files in state
    initial_state = {
        "uploaded_files": [
            {
                "name": "project_plan.pdf",
                "type": "PDF",
                "summary": "Q4 project timeline and milestones"
            },
            {
                "name": "budget_report.xlsx", 
                "type": "Excel",
                "summary": "Monthly budget breakdown and expenses"
            }
        ]
    }
    
    print("\n--- Querying with file context ---")
    context = ComplianceContext(
        user_id="user1",
        user_jurisdiction="US",
        industry="tech",
        compliance_frameworks=[]
    )
    
    # The inject_file_context middleware will add file info to the prompt
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "What files do I have and what's in the project plan?"}],
            **initial_state
        },
        context=context
    )

def demo_writing_style_injection():
    """Demo injecting user's writing style from store."""
    print("\n✍️ Writing Style Injection")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Set up writing styles for different users
    store.put(("writing_style",), "formal_user", {
        "tone": "professional",
        "greeting": "Dear colleague",
        "sign_off": "Best regards",
        "example": "I hope this message finds you well..."
    })
    
    store.put(("writing_style",), "casual_user", {
        "tone": "casual",
        "greeting": "Hey",
        "sign_off": "Cheers",
        "example": "Just wanted to quickly check in about..."
    })
    
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[inject_writing_style],
        context_schema=ComplianceContext,
        store=store,
        system_prompt="You are a writing assistant that adapts to user preferences."
    )
    
    # Formal user
    print("\n--- Formal user writing style ---")
    formal_context = ComplianceContext(
        user_id="formal_user",
        user_jurisdiction="US",
        industry="finance", 
        compliance_frameworks=[]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me write an email about the quarterly results"}]},
        context=formal_context
    )
    
    # Casual user
    print("\n--- Casual user writing style ---")
    casual_context = ComplianceContext(
        user_id="casual_user",
        user_jurisdiction="US",
        industry="tech",
        compliance_frameworks=[]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me write an email about the team meeting"}]},
        context=casual_context
    )

def demo_compliance_injection():
    """Demo injecting compliance rules based on context."""
    print("\n⚖️ Compliance Rules Injection")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[inject_compliance_rules],
        context_schema=ComplianceContext,
        system_prompt="You are a compliance-aware assistant."
    )
    
    # Healthcare with HIPAA
    print("\n--- Healthcare industry with HIPAA compliance ---")
    healthcare_context = ComplianceContext(
        user_id="doctor1",
        user_jurisdiction="US",
        industry="healthcare",
        compliance_frameworks=["HIPAA"]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Help me respond to a patient inquiry about their test results"}]},
        context=healthcare_context
    )
    
    # Finance with multiple frameworks
    print("\n--- Finance industry with multiple compliance frameworks ---")
    finance_context = ComplianceContext(
        user_id="advisor1", 
        user_jurisdiction="EU",
        industry="finance",
        compliance_frameworks=["GDPR", "MiFID"]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Draft investment advice for a client"}]},
        context=finance_context
    )

if __name__ == "__main__":
    print("💬 LangChain Message Context Injection Example")
    print("Shows how to modify messages before they reach the model\n")
    
    demo_file_context_injection()
    demo_writing_style_injection()
    demo_compliance_injection()
    
    print("\n✅ Message injection demo completed!")
    print("💬 Key concepts demonstrated:")
    print("   - @wrap_model_call middleware")
    print("   - File context injection from state")
    print("   - Writing style injection from store")
    print("   - Compliance rules injection from context")
    print("   - request.override(messages=...) modification")