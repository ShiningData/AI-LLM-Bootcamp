"""
Agentic RAG Implementation - Overview

This demonstrates agents that decide when and how to retrieve information.
Each topic has been split into focused, educational modules:

main5a_basic_agent.py          - Simple agent with retrieval tool
main5b_multi_source_retrieval.py - Multiple data sources and tool selection  
main5c_adaptive_retrieval.py     - Smart retrieval decisions based on context

Agentic RAG provides flexible, intelligent retrieval where agents reason about
when retrieval is needed and which sources to consult.

Run each file individually to learn specific aspects of agentic RAG systems.
"""

def print_overview():
    """Print an overview of all agentic RAG examples."""
    print("Agentic RAG Implementation - Complete Guide")
    print("=" * 52)
    print()
    
    examples = [
        {
            "file": "main5a_basic_agent.py",
            "title": "Basic Agent with Retrieval", 
            "description": "Simple agent setup with retrieval and calculation tools",
            "topics": [
                "Creating retrieval tools with @tool decorator",
                "Basic agent setup with LangChain agents",
                "Tool selection and invocation patterns",
                "Simple multi-tool reasoning workflows"
            ]
        },
        {
            "file": "main5b_multi_source_retrieval.py",
            "title": "Multi-Source Retrieval",
            "description": "Agents choosing between different data sources and tools",
            "topics": [
                "Multiple specialized retrieval tools",
                "Tool selection based on query type",
                "Combining technical docs, product info, and web sources",
                "Intelligent source routing strategies"
            ]
        },
        {
            "file": "main5c_adaptive_retrieval.py",
            "title": "Adaptive Retrieval Strategies",
            "description": "Smart agents that decide when retrieval is needed",
            "topics": [
                "Conditional tool usage based on context",
                "Efficient tool selection guidelines",
                "Avoiding unnecessary retrieval calls",
                "Context-aware response strategies"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   File: {example['file']}")
        print(f"   {example['description']}")
        print(f"   Topics covered:")
        for topic in example['topics']:
            print(f"   • {topic}")
        print()
    
    print("Getting Started:")
    print("   Run each example individually to focus on specific concepts:")
    print(f"   python main5a_basic_agent.py")
    print(f"   python main5b_multi_source_retrieval.py") 
    print(f"   python main5c_adaptive_retrieval.py")
    print()
    
    print("Quick Reference:")
    print("   • Agents: LLMs that can reason about tool usage")
    print("   • @tool: Decorator to create LangChain tools")
    print("   • Dynamic Retrieval: Agents decide when to retrieve")
    print("   • Multi-Source: Different tools for different data types")
    print("   • Adaptive Behavior: Context-aware tool selection")
    print()
    
    print("Best Practices:")
    print("   1. Start with main5a for basic agent concepts")
    print("   2. Create focused tools for specific data sources")
    print("   3. Use clear tool descriptions for agent reasoning")
    print("   4. Implement efficient tool selection logic")
    print("   5. Monitor tool usage patterns and optimize")
    print()

def demonstrate_quick_example():
    """Show a minimal quick-start example."""
    print("Quick Start Example")
    print("-" * 30)
    print()
    
    print("Here's the basic agentic RAG pattern:")
    print()
    
    code_example = '''# Agentic RAG Implementation
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# Initialize model
model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

@tool
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for information."""
    # Retrieval logic here
    docs = vectorstore.similarity_search(query, k=2)
    return "\\n".join([doc.page_content for doc in docs])

@tool
def calculate_pricing(plan: str, months: int) -> str:
    """Calculate pricing for a plan over time."""
    prices = {"basic": 29, "professional": 99}
    return f"{plan}: ${prices[plan] * months}"

# Create agent with tools
tools = [search_knowledge_base, calculate_pricing]
agent = create_agent(model, tools, system_prompt="Use tools to answer questions.")

# Agent decides when to use which tools
response = agent.invoke({"messages": [{"role": "user", "content": "What would 6 months of basic cost?"}]})'''
    
    print(code_example)
    print()
    print("Agents intelligently decide when and which tools to use!")
    print("   Explore the detailed examples to learn advanced patterns.")
    print()

def show_architecture_overview():
    """Show the agentic RAG architecture and patterns."""
    print("Agentic RAG Architecture Overview")
    print("-" * 38)
    print()
    
    print("Agentic Flow:")
    print("   1. User Query → Agent Reasoning")
    print("   2. Agent Reasoning → Tool Selection")
    print("   3. Tool Selection → Information Retrieval")
    print("   4. Information Retrieval → Result Processing") 
    print("   5. Result Processing → Response Generation")
    print("   6. Response Generation → Final Answer")
    print()
    
    print("Key Components:")
    print("   • Agent: LLM that reasons about tool usage")
    print("   • Tools: Retrieval functions with clear descriptions")
    print("   • Tool Selection: Agent chooses appropriate tools")
    print("   • Multi-Step Reasoning: Agents can chain tool calls")
    print("   • Dynamic Behavior: Adapts to different query types")
    print()
    
    print("Design Trade-offs:")
    print("   • Intelligence vs Control: Agents are flexible but less predictable")
    print("   • Latency vs Quality: Multiple tool calls take more time")
    print("   • Tool Complexity vs Capability: More tools = more options but harder to manage")
    print("   • Cost vs Performance: Agent reasoning adds token usage")
    print()

def show_comparison_with_other_patterns():
    """Compare agentic RAG with other RAG patterns."""
    print("RAG Pattern Comparison")
    print("-" * 30)
    print()
    
    patterns = {
        "Two-Step RAG": {
            "Retrieval": "Always happens",
            "Control": "High (predictable)",
            "Flexibility": "Low", 
            "Latency": "Consistent",
            "Best For": "FAQs, simple Q&A"
        },
        "Agentic RAG": {
            "Retrieval": "Agent decides when/how",
            "Control": "Low (adaptive)",
            "Flexibility": "High",
            "Latency": "Variable",
            "Best For": "Research, complex queries"
        },
        "Hybrid RAG": {
            "Retrieval": "Validation + agent decisions",
            "Control": "Medium",
            "Flexibility": "Medium-High",
            "Latency": "Variable", 
            "Best For": "Quality-critical applications"
        }
    }
    
    for pattern_name, characteristics in patterns.items():
        print(f"{pattern_name}:")
        for aspect, value in characteristics.items():
            print(f"   {aspect}: {value}")
        print()

if __name__ == "__main__":
    print_overview()
    demonstrate_quick_example()
    show_architecture_overview()
    show_comparison_with_other_patterns()
    
    print("Next Steps:")
    print("   • Run the individual examples to dive deeper")
    print("   • Start with main5a for basic agent setup")
    print("   • Use main5b for multi-source retrieval patterns")
    print("   • Apply main5c for adaptive retrieval strategies") 
    print("   • Move on to main6_hybrid_rag.py for validation-enhanced RAG")