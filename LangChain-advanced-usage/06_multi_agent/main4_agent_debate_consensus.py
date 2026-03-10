"""
Agent Debate and Consensus System.

Shows how to:
- Create agents with different perspectives that debate issues
- Implement structured argumentation and counter-argumentation
- Reach consensus through iterative discussion
- Handle conflicting viewpoints and find middle ground
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
    max_tokens=300
)

# Debate facilitation tools
@tool
def present_argument(position: str, supporting_evidence: str) -> str:
    """Present an argument with supporting evidence."""
    return f"ARGUMENT: {position}\nEVIDENCE: {supporting_evidence}\nStatus: Argument presented"

@tool
def counter_argument(original_position: str, counter_position: str, rebuttal_evidence: str) -> str:
    """Present a counter-argument to an existing position."""
    return f"COUNTER TO: {original_position}\nCOUNTER-POSITION: {counter_position}\nREBUTTAL: {rebuttal_evidence}\nStatus: Counter-argument presented"

@tool
def find_common_ground(position_a: str, position_b: str) -> str:
    """Identify areas of agreement between conflicting positions."""
    return f"COMMON GROUND ANALYSIS:\nPosition A: {position_a[:100]}...\nPosition B: {position_b[:100]}...\nAreas of agreement identified and documented"

@tool
def propose_compromise(conflicting_positions: str, compromise_solution: str) -> str:
    """Propose a compromise solution that addresses multiple positions."""
    return f"COMPROMISE PROPOSAL:\nConflicting positions considered: {conflicting_positions[:100]}...\nProposed solution: {compromise_solution}\nStatus: Compromise ready for evaluation"

class AgentDebateSystem:
    """Manages structured debates between agents with different perspectives."""
    
    def __init__(self):
        self.debate_history = []
        self.current_round = 0
        
    def create_advocate_agent(self, perspective: str, expertise: str):
        """Create an agent that advocates for a specific perspective."""
        return create_agent(
            model,
            tools=[present_argument, counter_argument],
            system_prompt=f"""You are a {expertise} expert who advocates for the {perspective} perspective.

Your role in debates:
- Use present_argument to make strong cases for your perspective
- Use counter_argument to challenge opposing viewpoints
- Back arguments with relevant evidence and expertise
- Stay focused on your assigned perspective while being respectful
- Acknowledge valid points from others but maintain your position

Your expertise: {expertise}
Your perspective: {perspective}

Be persuasive but professional in all arguments."""
        )
    
    def create_moderator_agent(self):
        """Create a moderator agent that facilitates consensus."""
        return create_agent(
            model,
            tools=[find_common_ground, propose_compromise],
            system_prompt="""You are a neutral moderator focused on facilitating productive debate and finding consensus.

Your role:
- Use find_common_ground to identify areas of agreement
- Use propose_compromise to suggest balanced solutions
- Synthesize multiple viewpoints into workable solutions
- Ensure all perspectives are fairly considered
- Guide the debate toward constructive outcomes

Stay neutral and focus on finding practical solutions that address legitimate concerns from all sides."""
        )
    
    def conduct_debate(self, topic: str, agents_config: list, max_rounds: int = 3):
        """Conduct a structured debate on a given topic."""
        print(f"🗣️ DEBATE: {topic}")
        print("=" * 60)
        
        # Create agents based on configuration
        agents = {}
        moderator = self.create_moderator_agent()
        
        for name, perspective, expertise in agents_config:
            agents[name] = self.create_advocate_agent(perspective, expertise)
        
        debate_results = []
        
        # Round 1: Initial positions
        print(f"\n--- ROUND 1: INITIAL POSITIONS ---")
        for name, agent in agents.items():
            print(f"\n{name} presenting initial position:")
            result = agent.invoke({
                "messages": [{"role": "user", "content": f"Present your initial position on: {topic}. Make a strong argument for your perspective."}]
            })
            position = extract_clean_content(result['messages'][-1].content)
            debate_results.append(f"{name}: {position}")
            print(f"{position[:200]}...")
        
        # Round 2: Counter-arguments
        print(f"\n--- ROUND 2: COUNTER-ARGUMENTS ---")
        agent_names = list(agents.keys())
        for i, (name, agent) in enumerate(agents.items()):
            # Get opponent's position to counter
            opponent_name = agent_names[(i + 1) % len(agent_names)]
            opponent_full = [r for r in debate_results if r.startswith(opponent_name + ":")][0]
            opponent_position = opponent_full.split(":", 1)[1].strip()  # Remove agent name prefix
            
            print(f"\n{name} presenting counter-argument:")
            result = agent.invoke({
                "messages": [{"role": "user", "content": f"Present a counter-argument to {opponent_name}'s position: '{opponent_position[:200]}...'\n\nChallenge their reasoning while supporting your own perspective."}]
            })
            counter_arg = extract_clean_content(result['messages'][-1].content) 
            debate_results.append(f"{name} counters: {counter_arg}")
            print(f"{counter_arg[:200]}...")
        
        # Round 3: Moderated consensus seeking
        print(f"\n--- ROUND 3: SEEKING CONSENSUS ---")
        
        # Compile all debate points
        all_positions = "\n\n".join(debate_results)
        
        print(f"\nModerator analyzing all positions...")
        consensus_result = moderator.invoke({
            "messages": [{"role": "user", "content": f"Moderate this debate on '{topic}' and help find consensus. Here are all the positions presented:\n\n{all_positions}\n\nFind common ground and propose a compromise solution that addresses the main concerns of all parties."}]
        })
        
        consensus = extract_clean_content(consensus_result['messages'][-1].content)
        print(f"\nMODERATOR CONSENSUS:\n{consensus}")
        
        return {
            "topic": topic,
            "debate_results": debate_results,
            "consensus": consensus
        }

def demonstrate_policy_debate():
    """Demonstrate debate on a policy issue."""
    print("=== Policy Debate Example ===\n")
    
    debate_system = AgentDebateSystem()
    
    # Configure agents with opposing perspectives
    agents_config = [
        ("Privacy_Advocate", "privacy protection", "privacy law and digital rights"),
        ("Innovation_Supporter", "technological progress", "AI technology and business development"),
        ("Ethics_Expert", "ethical AI development", "AI ethics and responsible technology")
    ]
    
    topic = "Should AI companies be required to disclose their training data sources?"
    
    result = debate_system.conduct_debate(topic, agents_config)
    
    print("\n" + "="*60)
    print("DEBATE SUMMARY:")
    print(f"Topic: {result['topic']}")
    print(f"Positions presented: {len(result['debate_results'])}")
    print(f"Consensus reached: {'Yes' if 'compromise' in result['consensus'].lower() or 'solution' in result['consensus'].lower() else 'Partial'}")

def demonstrate_technical_debate():
    """Demonstrate debate on a technical issue."""
    print("\n=== Technical Debate Example ===\n")
    
    debate_system = AgentDebateSystem()
    
    # Configure technical experts with different approaches
    agents_config = [
        ("Performance_Expert", "performance optimization", "system performance and scalability"),
        ("Security_Expert", "security-first approach", "cybersecurity and system hardening"),
    ]
    
    topic = "Should we prioritize performance or security in our new API design?"
    
    result = debate_system.conduct_debate(topic, agents_config, max_rounds=2)
    
    print("\n" + "="*60) 
    print("TECHNICAL DEBATE SUMMARY:")
    print(f"Topic: {result['topic']}")
    print(f"Expert perspectives: {len([r for r in result['debate_results'] if not r.startswith('Moderator')])}")

def demonstrate_consensus_building():
    """Show how the system builds consensus from conflict."""
    print("\n=== Consensus Building Process ===\n")
    
    debate_system = AgentDebateSystem()
    
    # Three-way debate with conflicting priorities
    agents_config = [
        ("Cost_Controller", "cost minimization", "financial management and budget optimization"),
        ("Quality_Advocate", "quality maximization", "quality assurance and customer satisfaction"),
        ("Speed_Optimizer", "fast delivery", "project management and rapid deployment")
    ]
    
    topic = "How should we balance cost, quality, and speed in our product development?"
    
    result = debate_system.conduct_debate(topic, agents_config)
    
    print(f"\nCONSENSUS QUALITY INDICATORS:")
    consensus_text = result['consensus'].lower()
    print(f"✅ Addresses multiple perspectives: {'balance' in consensus_text or 'all' in consensus_text}")
    print(f"✅ Proposes actionable solution: {'should' in consensus_text or 'recommend' in consensus_text}")
    print(f"✅ Acknowledges trade-offs: {'trade' in consensus_text or 'compromise' in consensus_text}")

if __name__ == "__main__":
    print("🗣️ Agent Debate and Consensus System")
    print("This system facilitates structured debates between agents to reach informed consensus\n")
    
    demonstrate_policy_debate()
    demonstrate_technical_debate()
    demonstrate_consensus_building()
    
    print("\n✅ Debate System Benefits:")
    print("   🎯 Multiple perspective consideration")
    print("   🗣️ Structured argumentation process")
    print("   🤝 Consensus-driven solutions")
    print("   ⚖️ Balanced decision making")
    print("   📝 Documented reasoning process")