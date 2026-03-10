"""
Tool Calling Multi-Agent Pattern.

Shows how to:
- Create a supervisor agent that calls other agents as tools
- Implement specialized agent tools (researcher, analyzer, planner)
- Use centralized control flow for task orchestration
- Handle complex tasks by breaking them into specialized subtasks
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

def extract_clean_content(message_content):
    """Extract clean text content from message, handling both string and structured content."""
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        # Handle structured content - extract text only
        text_parts = []
        for item in message_content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            elif isinstance(item, dict) and 'type' in item and item['type'] == 'text':
                text_parts.append(item.get('text', ''))
        return ' '.join(text_parts) if text_parts else str(message_content)
    else:
        return str(message_content)

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

# Create specialized agent tools
@tool
def research_agent(topic: str) -> str:
    """Research agent that gathers information on a given topic."""
    # Simulate research results
    research_results = {
        "ai": "AI research shows rapid growth in LLMs, computer vision, and robotics. Key trends include multimodal models and efficiency improvements.",
        "climate": "Climate research indicates accelerating warming trends, renewable energy adoption, and carbon capture technologies as key focus areas.",
        "default": f"Research on {topic}: Found comprehensive information about current trends, challenges, and opportunities in this field."
    }
    return research_results.get(topic.lower(), research_results["default"])

@tool  
def analysis_agent(data: str) -> str:
    """Analysis agent that processes and analyzes data or information."""
    analysis = f"""
Analysis Results:
- Key themes identified: {len(data.split())} concepts analyzed
- Complexity level: {"High" if len(data) > 200 else "Medium" if len(data) > 100 else "Low"}
- Main insights: The data shows clear patterns and actionable information
- Recommendations: Focus on the most impactful findings for decision-making
"""
    return analysis

@tool
def planning_agent(requirements: str) -> str:
    """Planning agent that creates structured plans and strategies."""
    plan = f"""
Strategic Plan Based on Requirements:
1. Assessment Phase
   - Review current situation and requirements
   - Identify key stakeholders and resources

2. Development Phase  
   - Design solution approach
   - Allocate timeline and milestones

3. Implementation Phase
   - Execute planned activities
   - Monitor progress and adjust as needed

4. Evaluation Phase
   - Measure outcomes against goals
   - Document lessons learned

Requirements addressed: {requirements[:100]}...
"""
    return plan

@tool
def coordination_agent(task_results: str) -> str:
    """Coordination agent that synthesizes results from multiple agents."""
    synthesis = f"""
Coordination Summary:
- Multiple agent inputs processed
- Cross-functional insights integrated  
- Unified recommendations prepared
- Next steps clearly defined

Synthesized from: {len(task_results.split())} data points
Status: Ready for final decision-making
"""
    return synthesis

def demonstrate_tool_calling_pattern():
    """Demonstrate the tool calling multi-agent pattern."""
    print("=== Tool Calling Multi-Agent Pattern ===\n")
    
    # Create supervisor agent with other agents as tools
    supervisor_agent = create_agent(
        model,
        tools=[research_agent, analysis_agent, planning_agent, coordination_agent],
        system_prompt="""You are a supervisor agent that coordinates specialized agents to solve complex problems.

Your available agents:
- research_agent: Gathers information on topics
- analysis_agent: Analyzes data and information  
- planning_agent: Creates strategic plans
- coordination_agent: Synthesizes results from multiple agents

For complex requests:
1. Break down the task into specialized subtasks
2. Call the appropriate agents in logical order
3. Use coordination_agent to synthesize results when multiple agents are used
4. Provide a clear summary of the complete solution

Always use the most appropriate agents for each subtask and coordinate their outputs effectively."""
    )

    # Test case 1: Simple single-agent task
    print("--- Test 1: Single Agent Task ---")
    result = supervisor_agent.invoke({
        "messages": [{"role": "user", "content": "Research current trends in artificial intelligence"}]
    })
    clean_content = extract_clean_content(result['messages'][-1].content)
    print(f"Response: {clean_content[:300]}...\n")

    # Test case 2: Multi-agent coordination task  
    print("--- Test 2: Multi-Agent Coordination Task ---")
    result = supervisor_agent.invoke({
        "messages": [{"role": "user", "content": "I need to develop a strategy for implementing AI in our company. Research the field, analyze our options, and create a comprehensive plan."}]
    })
    clean_content = extract_clean_content(result['messages'][-1].content)
    print(f"Response: {clean_content[:400]}...\n")

    # Test case 3: Complex synthesis task
    print("--- Test 3: Complex Synthesis Task ---")
    result = supervisor_agent.invoke({
        "messages": [{"role": "user", "content": "Research climate change solutions, analyze the most promising approaches, plan an implementation strategy, and coordinate everything into a unified recommendation."}]
    })
    clean_content = extract_clean_content(result['messages'][-1].content)
    print(f"Response: {clean_content[:400]}...\n")

def demonstrate_agent_specialization():
    """Show how each specialized agent works independently."""
    print("=== Individual Agent Capabilities ===\n")
    
    # Test each agent individually to show their specialization
    print("--- Research Agent Specialization ---")
    research_result = research_agent.invoke({"topic": "AI"})
    print(f"Research output: {research_result}\n")
    
    print("--- Analysis Agent Specialization ---") 
    analysis_result = analysis_agent.invoke({"data": "Large dataset with multiple trends and patterns showing significant growth in technology adoption"})
    print(f"Analysis output: {analysis_result}\n")
    
    print("--- Planning Agent Specialization ---")
    planning_result = planning_agent.invoke({"requirements": "Implement new customer service chatbot with 24/7 availability"})
    print(f"Planning output: {planning_result[:200]}...\n")

if __name__ == "__main__":
    print("🤖 Tool Calling Multi-Agent Pattern Example")
    print("This pattern uses a supervisor agent that calls specialized agents as tools\n")
    
    demonstrate_tool_calling_pattern()
    demonstrate_agent_specialization()
    
    print("✅ Tool Calling Pattern Benefits:")
    print("   🎯 Centralized control and coordination")
    print("   🔧 Specialized agent capabilities")  
    print("   📋 Structured task breakdown")
    print("   🔄 Reusable agent components")
    print("   🛡️ Predictable execution flow")