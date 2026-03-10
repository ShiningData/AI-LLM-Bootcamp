"""
Collaborative Multi-Agent Research System.

Shows how to:
- Create agents that collaborate on research tasks
- Implement parallel information gathering
- Synthesize results from multiple research perspectives  
- Handle complex multi-step research workflows
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
import time

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
    max_tokens=400
)

# Research-specific tools
@tool
def academic_search(topic: str) -> str:
    """Search academic papers and journals for research topic."""
    # Simulate academic research
    academic_results = {
        "machine learning": "Academic research shows ML advances in transformer architectures, few-shot learning, and interpretability. 15+ papers from 2024.",
        "climate change": "Recent academic work focuses on carbon capture, renewable integration, and climate modeling. 20+ peer-reviewed studies.",
        "quantum computing": "Academic progress in quantum error correction, NISQ algorithms, and quantum advantage proofs. 10+ research papers.",
        "default": f"Academic search for '{topic}': Found 12 peer-reviewed papers with theoretical foundations and experimental validation."
    }
    return academic_results.get(topic.lower(), academic_results["default"])

@tool
def industry_trends_search(topic: str) -> str:
    """Search for current industry trends and market analysis."""
    # Simulate industry research
    industry_results = {
        "machine learning": "Industry trends: MLOps adoption, edge AI deployment, generative AI integration. Market size: $96B by 2025.",
        "climate change": "Industry focus: ESG compliance, carbon trading, clean tech investments. $2.5T climate tech market emerging.",
        "quantum computing": "Industry developments: IBM quantum roadmap, Google quantum supremacy, commercial applications in optimization.",
        "default": f"Industry analysis for '{topic}': Current market trends, key players, and growth projections identified."
    }
    return industry_results.get(topic.lower(), industry_results["default"])

@tool
def news_analysis(topic: str) -> str:
    """Analyze recent news and current events related to topic."""
    # Simulate news analysis
    news_results = {
        "machine learning": "Recent news: ChatGPT-4 updates, Google Bard improvements, AI regulation discussions. 50+ articles this month.",
        "climate change": "Recent developments: COP28 outcomes, renewable energy records, climate policy updates. High media coverage.",
        "quantum computing": "Latest news: Quantum computing breakthroughs, IBM partnerships, venture capital funding. Growing interest.",
        "default": f"News analysis for '{topic}': 25+ recent articles covering developments, controversies, and expert opinions."
    }
    return news_results.get(topic.lower(), news_results["default"])

@tool
def expert_opinions_search(topic: str) -> str:
    """Gather expert opinions and thought leadership content."""
    # Simulate expert opinion gathering
    expert_results = {
        "machine learning": "Expert consensus: Transformers remain dominant, multimodal AI emerging, ethical considerations paramount. 10+ thought leaders.",
        "climate change": "Expert views: Urgent action needed, technology solutions promising, policy coordination critical. Leading scientists concur.",
        "quantum computing": "Expert opinions: NISQ era continuing, fault-tolerant systems 5-10 years away, commercial applications limited currently.",
        "default": f"Expert opinions on '{topic}': Diverse perspectives from 8+ industry leaders and academic experts collected."
    }
    return expert_results.get(topic.lower(), expert_results["default"])

class CollaborativeResearchSystem:
    """Manages collaborative research between specialized agents."""
    
    def __init__(self):
        self.researchers = {
            "academic": self._create_academic_researcher(),
            "industry": self._create_industry_researcher(),
            "news": self._create_news_analyst(),
            "synthesis": self._create_synthesis_agent()
        }
        self.research_results = {}
    
    def _create_academic_researcher(self):
        """Create academic research specialist."""
        return create_agent(
            model,
            tools=[academic_search, expert_opinions_search],
            system_prompt="""You are an academic research specialist focused on scholarly sources.

Your role:
- Use academic_search for peer-reviewed research
- Use expert_opinions_search for academic expert views
- Focus on theoretical foundations and empirical evidence
- Identify research gaps and future directions
- Provide citations and academic credibility assessment

Present findings in academic style with clear evidence basis."""
        )
    
    def _create_industry_researcher(self):
        """Create industry analysis specialist."""
        return create_agent(
            model,
            tools=[industry_trends_search, expert_opinions_search],
            system_prompt="""You are an industry research specialist focused on market analysis.

Your role:
- Use industry_trends_search for market data and business trends
- Use expert_opinions_search for industry leader perspectives  
- Focus on commercial applications and market opportunities
- Identify business drivers and competitive landscape
- Assess practical implementation feasibility

Present findings with business and market focus."""
        )
    
    def _create_news_analyst(self):
        """Create current events analyst."""
        return create_agent(
            model,
            tools=[news_analysis, expert_opinions_search],
            system_prompt="""You are a news analyst focused on current developments.

Your role:
- Use news_analysis for recent events and developments
- Use expert_opinions_search for current expert commentary
- Focus on trending topics and emerging issues
- Identify public sentiment and media coverage patterns
- Track policy and regulatory developments

Present findings highlighting current relevance and trends."""
        )
    
    def _create_synthesis_agent(self):
        """Create research synthesis specialist."""
        return create_agent(
            model,
            tools=[],
            system_prompt="""You are a research synthesis specialist who combines findings from multiple researchers.

Your role:
- Analyze findings from academic, industry, and news researchers
- Identify convergent themes and conflicting viewpoints
- Synthesize comprehensive overview with multiple perspectives
- Highlight key insights and actionable recommendations
- Create balanced assessment of current state and future outlook

Present synthesized findings as comprehensive research summary."""
        )
    
    def conduct_collaborative_research(self, topic: str) -> dict:
        """Conduct collaborative research with multiple specialized agents."""
        print(f"🔬 Starting collaborative research on: {topic}\n")
        
        # Phase 1: Parallel research by specialists
        research_tasks = [
            ("academic", "Academic Research"),
            ("industry", "Industry Analysis"), 
            ("news", "Current Events Analysis")
        ]
        
        results = {}
        
        for researcher_type, description in research_tasks:
            print(f"--- {description} ---")
            
            researcher = self.researchers[researcher_type]
            result = researcher.invoke({
                "messages": [{"role": "user", "content": f"Research {topic} from your specialized perspective. Provide comprehensive findings."}]
            })
            
            findings = extract_clean_content(result['messages'][-1].content)
            results[researcher_type] = findings
            
            print(f"{description} Complete: {len(findings)} chars")
            print(f"Key findings: {findings[:150]}...\n")
        
        # Phase 2: Synthesis of all findings
        print("--- Research Synthesis ---")
        
        synthesis_prompt = f"""Synthesize research findings on '{topic}' from multiple perspectives:

ACADEMIC FINDINGS:
{results['academic']}

INDUSTRY FINDINGS:  
{results['industry']}

NEWS ANALYSIS:
{results['news']}

Please provide a comprehensive synthesis that:
1. Identifies key convergent themes across all sources
2. Notes any conflicting viewpoints or discrepancies  
3. Highlights the most significant insights
4. Provides actionable recommendations
5. Assesses current state and future outlook"""

        synthesizer = self.researchers['synthesis']
        synthesis_result = synthesizer.invoke({
            "messages": [{"role": "user", "content": synthesis_prompt}]
        })
        
        synthesis = extract_clean_content(synthesis_result['messages'][-1].content)
        results['synthesis'] = synthesis
        
        print(f"Research Synthesis Complete: {len(synthesis)} chars")
        print(f"Key synthesis: {synthesis[:200]}...\n")
        
        return results

def demonstrate_collaborative_research():
    """Demonstrate collaborative research system."""
    print("=== Collaborative Multi-Agent Research ===\n")
    
    research_system = CollaborativeResearchSystem()
    
    # Test topics
    topics = [
        "machine learning",
        "quantum computing"
    ]
    
    for topic in topics:
        print(f"{'='*60}")
        print(f"RESEARCH PROJECT: {topic.upper()}")
        print(f"{'='*60}")
        
        results = research_system.conduct_collaborative_research(topic)
        
        print("📋 FINAL RESEARCH SUMMARY:")
        print(f"✅ Academic research: {len(results['academic'])} chars")
        print(f"✅ Industry analysis: {len(results['industry'])} chars") 
        print(f"✅ News analysis: {len(results['news'])} chars")
        print(f"✅ Synthesis: {len(results['synthesis'])} chars")
        print()

def demonstrate_individual_researchers():
    """Show how each researcher specializes in different aspects."""
    print("=== Individual Researcher Specializations ===\n")
    
    system = CollaborativeResearchSystem()
    topic = "artificial intelligence ethics"
    
    for researcher_type, researcher in system.researchers.items():
        if researcher_type == "synthesis":
            continue
            
        print(f"--- {researcher_type.title()} Researcher ---")
        result = researcher.invoke({
            "messages": [{"role": "user", "content": f"Research {topic} from your specialized perspective"}]
        })
        
        findings = extract_clean_content(result['messages'][-1].content)
        print(f"Specialization: {findings[:250]}...\n")

if __name__ == "__main__":
    print("🔬 Collaborative Multi-Agent Research System")
    print("This system combines multiple research perspectives for comprehensive analysis\n")
    
    demonstrate_collaborative_research()
    demonstrate_individual_researchers()
    
    print("✅ Collaborative Research Benefits:")
    print("   🎯 Multiple specialized perspectives")
    print("   📚 Comprehensive information coverage")
    print("   🔍 Parallel research efficiency") 
    print("   🧩 Synthesized unified insights")
    print("   📊 Balanced academic/industry/current view")