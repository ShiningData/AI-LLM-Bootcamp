"""
Handoffs Multi-Agent Pattern.

Shows how to:
- Create agents that can transfer control to each other
- Implement decentralized agent-to-agent handoffs
- Handle dynamic conversation routing based on expertise
- Manage state transfer between active agents
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
    max_tokens=400
)

# Shared tools for handoff communication
@tool
def handoff_to_technical_specialist(context: str) -> str:
    """Transfer control to technical specialist agent for technical questions."""
    return f"HANDOFF: Transferring to technical specialist with context: {context}"

@tool
def handoff_to_sales_specialist(context: str) -> str:
    """Transfer control to sales specialist agent for pricing and business questions."""
    return f"HANDOFF: Transferring to sales specialist with context: {context}"

@tool  
def handoff_to_support_specialist(context: str) -> str:
    """Transfer control to support specialist agent for customer service issues."""
    return f"HANDOFF: Transferring to support specialist with context: {context}"

@tool
def handoff_to_general_assistant(context: str) -> str:
    """Transfer control back to general assistant for non-specialized tasks."""
    return f"HANDOFF: Transferring to general assistant with context: {context}"

# Domain-specific tools
@tool
def check_technical_docs(query: str) -> str:
    """Search technical documentation for implementation details."""
    return f"Technical docs for '{query}': Found API references, code examples, and implementation guides."

@tool
def get_pricing_info(product: str) -> str:
    """Get current pricing information for products."""
    pricing_data = {
        "basic": "$29/month - Basic features, 1000 API calls",
        "professional": "$99/month - Advanced features, 10000 API calls", 
        "enterprise": "$299/month - Full features, unlimited API calls"
    }
    return pricing_data.get(product.lower(), f"Pricing for {product}: Contact sales for custom quote")

@tool
def create_support_ticket(issue: str) -> str:
    """Create a support ticket for customer issues."""
    return f"Support ticket created for: {issue}. Ticket #SUP-{hash(issue) % 10000}. Our team will respond within 24 hours."

class MultiAgentHandoffSystem:
    """Manages handoffs between specialized agents."""
    
    def __init__(self):
        self.current_agent = "general"
        self.conversation_history = []
        
        # Create specialized agents
        self.agents = {
            "general": self._create_general_agent(),
            "technical": self._create_technical_agent(), 
            "sales": self._create_sales_agent(),
            "support": self._create_support_agent()
        }
    
    def _create_general_agent(self):
        """Create general purpose agent that can handoff to specialists."""
        return create_agent(
            model,
            tools=[handoff_to_technical_specialist, handoff_to_sales_specialist, handoff_to_support_specialist],
            system_prompt="""You are a general assistant that helps route conversations to specialists.

Analyze user requests and decide if you need to handoff:
- For technical questions (APIs, code, implementation): Use handoff_to_technical_specialist  
- For pricing, sales, business questions: Use handoff_to_sales_specialist
- For customer support, issues, complaints: Use handoff_to_support_specialist

When handing off, provide clear context about what the user needs.
For simple general questions, answer directly without handoff."""
        )
    
    def _create_technical_agent(self):
        """Create technical specialist agent."""
        return create_agent(
            model,
            tools=[check_technical_docs, handoff_to_general_assistant, handoff_to_support_specialist],
            system_prompt="""You are a technical specialist focused on implementation, APIs, and code.

Use check_technical_docs to find specific technical information.
Provide detailed technical guidance and code examples.
If the question becomes non-technical, handoff to general_assistant.
For customer issues, handoff to support_specialist."""
        )
    
    def _create_sales_agent(self):
        """Create sales specialist agent."""
        return create_agent(
            model, 
            tools=[get_pricing_info, handoff_to_general_assistant, handoff_to_technical_specialist],
            system_prompt="""You are a sales specialist focused on pricing, plans, and business value.

Use get_pricing_info to provide accurate pricing details.
Focus on business benefits and ROI.
For technical implementation questions, handoff to technical_specialist.
For general questions, handoff to general_assistant."""
        )
    
    def _create_support_agent(self):
        """Create customer support specialist agent."""
        return create_agent(
            model,
            tools=[create_support_ticket, handoff_to_technical_specialist, handoff_to_general_assistant],
            system_prompt="""You are a customer support specialist focused on resolving issues.

Use create_support_ticket for problems that need follow-up.
Provide helpful troubleshooting and solutions.
For technical details, handoff to technical_specialist.
For general questions, handoff to general_assistant."""
        )
    
    def process_message(self, message: str) -> str:
        """Process a message with the current active agent."""
        # Get current agent
        agent = self.agents[self.current_agent]
        
        # Process the message
        result = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        
        response_content = extract_clean_content(result['messages'][-1].content)
        
        # Check for handoff in the response
        if "HANDOFF:" in response_content:
            self._handle_handoff(response_content)
            # Process with new agent
            new_agent = self.agents[self.current_agent]
            result = new_agent.invoke({
                "messages": [{"role": "user", "content": message}]
            })
            response_content = extract_clean_content(result['messages'][-1].content)
        
        return response_content
    
    def _handle_handoff(self, response: str):
        """Handle agent handoff based on response."""
        if "technical specialist" in response:
            self.current_agent = "technical"
        elif "sales specialist" in response:
            self.current_agent = "sales"
        elif "support specialist" in response:
            self.current_agent = "support"
        elif "general assistant" in response:
            self.current_agent = "general"

def demonstrate_handoffs_pattern():
    """Demonstrate the handoffs multi-agent pattern."""
    print("=== Handoffs Multi-Agent Pattern ===\n")
    
    # Create the handoff system
    system = MultiAgentHandoffSystem()
    
    # Test various conversation flows
    test_messages = [
        "Hi, I'm interested in your product. What does it cost?",
        "How do I implement the REST API in Python?", 
        "I'm having trouble with my account login",
        "What's the weather like today?",
        "Can you show me a code example for authentication?",
        "What's included in the enterprise plan?",
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"--- Test {i}: {message} ---")
        print(f"Active Agent: {system.current_agent}")
        
        response = system.process_message(message)
        print(f"Response: {response[:300]}...")
        
        print(f"New Active Agent: {system.current_agent}\n")

def demonstrate_agent_specialization():
    """Show how different agents handle the same query differently."""
    print("=== Agent Specialization Comparison ===\n")
    
    system = MultiAgentHandoffSystem()
    query = "Tell me about your API rate limits"
    
    for agent_type in ["general", "technical", "sales", "support"]:
        print(f"--- {agent_type.title()} Agent Response ---")
        system.current_agent = agent_type
        response = system.process_message(query)
        print(f"Response: {response[:200]}...\n")

if __name__ == "__main__":
    print("🔄 Handoffs Multi-Agent Pattern Example")
    print("This pattern allows agents to transfer control dynamically based on expertise\n")
    
    demonstrate_handoffs_pattern()
    demonstrate_agent_specialization()
    
    print("✅ Handoffs Pattern Benefits:")
    print("   🎯 Dynamic expertise routing")
    print("   🔄 Flexible conversation flow")
    print("   🎭 Specialized agent personalities") 
    print("   🗣️ Direct agent-user interaction")
    print("   📈 Scalable specialist addition")