"""
Dynamic prompts with runtime context example.

Shows how to:
- Create dynamic system prompts based on runtime context
- Use @dynamic_prompt decorator
- Access ModelRequest.runtime in middleware
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class PersonalityContext:
    user_name: str
    personality_type: str  # formal, casual, technical, friendly
    expertise_level: str   # beginner, intermediate, expert

# Dynamic prompt that adapts to user context
@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user personality and expertise."""
    ctx = request.runtime.context
    
    # Customize tone based on personality
    if ctx.personality_type == "formal":
        tone = "professional and respectful"
    elif ctx.personality_type == "casual":
        tone = "relaxed and friendly"
    elif ctx.personality_type == "technical":
        tone = "precise and technical"
    else:
        tone = "warm and encouraging"
    
    # Customize complexity based on expertise
    if ctx.expertise_level == "beginner":
        complexity = "Use simple terms and provide detailed explanations."
    elif ctx.expertise_level == "expert":
        complexity = "Use technical language and be concise."
    else:
        complexity = "Use moderate technical detail with examples."
    
    return f"""You are a {tone} assistant helping {ctx.user_name}.
{complexity}
Adapt your communication style to match the user's preferences."""

@tool
def explain_concept(concept: str, runtime: ToolRuntime[PersonalityContext]) -> str:
    """Explain a concept tailored to the user's expertise level."""
    ctx = runtime.context
    level = ctx.expertise_level
    
    if level == "beginner":
        detail = "basic explanation with examples"
    elif level == "expert":
        detail = "advanced technical details"
    else:
        detail = "moderate detail with practical applications"
    
    return f"Explaining '{concept}' with {detail} for {ctx.user_name}"

def demo_dynamic_prompts():
    """Demo dynamic prompt generation."""
    print("🎨 Dynamic Prompts Demo")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[explain_concept],
        middleware=[adaptive_system_prompt],
        context_schema=PersonalityContext,
        system_prompt="Default prompt (will be replaced by dynamic prompt)"
    )
    
    # Test different personality/expertise combinations
    test_contexts = [
        PersonalityContext(
            user_name="Dr. Smith",
            personality_type="formal", 
            expertise_level="expert"
        ),
        PersonalityContext(
            user_name="Jake",
            personality_type="casual",
            expertise_level="beginner"  
        ),
        PersonalityContext(
            user_name="Sarah",
            personality_type="technical",
            expertise_level="intermediate"
        )
    ]
    
    for ctx in test_contexts:
        print(f"\n--- Dynamic prompt for {ctx.user_name} ---")
        print(f"Type: {ctx.personality_type}, Level: {ctx.expertise_level}")
        
        agent.invoke(
            {"messages": [{"role": "user", "content": "Explain machine learning concepts"}]},
            context=ctx
        )
        print(f"✅ Prompt adapted for {ctx.user_name}")

if __name__ == "__main__":
    print("🎨 LangChain Dynamic Prompts Example")
    print("Shows how to generate personalized system prompts\n")
    
    demo_dynamic_prompts()
    
    print("\n✅ Dynamic prompts demo completed!")
    print("🎨 Key concepts demonstrated:")
    print("   - @dynamic_prompt decorator")
    print("   - ModelRequest.runtime access")
    print("   - Context-based prompt customization")
    print("   - Personality and expertise adaptation")