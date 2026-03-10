"""
Structured output example with LangChain agents.

In this example you will see:
- How to define structured output schemas with Pydantic BaseModel
- How to use automatic schema selection for structured output
- How to use ToolStrategy for explicit structured output control
- How to customize tool message content in structured responses
- How structured validation and error handling works
"""
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def get_additional_info(query: str) -> str:
    """Get additional information if needed."""
    return f"Additional info about: {query}"

# Example 1: Contact Information Extraction
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")  
    phone: str = Field(description="The phone number of the person")

# Example 2: Product Review Analysis
class ProductReview(BaseModel):
    """Analysis of a product review."""
    # Rating with validation constraints
    rating: int | None = Field(
        description="The rating of the product (1-5)",
        ge=1,   # minimum value = 1
        le=5    # maximum value = 5
    )
    # Sentiment with restricted values
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    # List of key points
    key_points: list[str] = Field(
        description="Key points from the review. Lowercase, 1-3 words each."
    )

# Example 3: Meeting Action Items  
class MeetingAction(BaseModel):
    """Action items extracted from meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")

def demo_automatic_schema():
    """Demo: Automatic schema selection for structured output."""
    print("🚀 DEMO 1: Automatic Schema Selection")
    print("=" * 60)
    
    # Create agent with automatic schema selection
    # LangChain automatically chooses the best strategy based on model capabilities
    agent = create_agent(
        model=model,
        tools=[get_additional_info],
        response_format=ContactInfo,  # Just pass the schema type
        system_prompt="Extract contact information from the provided text."
    )
    
    # Test the agent
    result = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Extract contact info: Jane Smith, jane.smith@company.com, +1-555-987-6543"
        }]
    })
    
    # Access structured output
    contact = result["structured_response"]
    print(f"📋 Extracted Contact Info:")
    print(f"   Name: {contact.name}")
    print(f"   Email: {contact.email}")  
    print(f"   Phone: {contact.phone}")
    print(f"   Type: {type(contact)}")
    print()

def demo_tool_strategy():
    """Demo: Explicit ToolStrategy for structured output."""
    print("🚀 DEMO 2: ToolStrategy with Product Review Analysis")
    print("=" * 60)
    
    # Create agent with explicit ToolStrategy
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(schema=ProductReview),
        system_prompt="Analyze product reviews and extract structured data."
    )
    
    # Test with a product review
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Analyze this review: 'Amazing product! 5 stars. Quick delivery and great quality, but quite pricey.'"
        }]
    })
    
    # Access structured output
    review = result["structured_response"]
    print(f"📊 Product Review Analysis:")
    print(f"   Rating: {review.rating}/5")
    print(f"   Sentiment: {review.sentiment}")
    print(f"   Key Points: {review.key_points}")
    print(f"   Type: {type(review)}")
    print()

def demo_custom_tool_message():
    """Demo: Custom tool message content in structured output."""
    print("🚀 DEMO 3: Custom Tool Message Content")
    print("=" * 60)
    
    # Create agent with custom tool message content
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(
            schema=MeetingAction,
            tool_message_content="✅ Action item captured and added to meeting notes!"
        ),
        system_prompt="Extract action items from meeting transcripts."
    )
    
    # Test with meeting notes
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "From our meeting: Bob needs to prepare the quarterly report by Friday - high priority"
        }]
    })
    
    # Access structured output
    action = result["structured_response"]
    print(f"📝 Meeting Action Item:")
    print(f"   Task: {action.task}")
    print(f"   Assignee: {action.assignee}")
    print(f"   Priority: {action.priority}")
    print(f"   Type: {type(action)}")
    
    # Show the custom tool message in conversation history
    print(f"\n💬 Last message in conversation:")
    last_message = result["messages"][-1]
    if hasattr(last_message, 'content'):
        print(f"   Content: {last_message.content}")
    print()

def demo_validation_handling():
    """Demo: How structured output handles validation."""
    print("🚀 DEMO 4: Validation and Error Handling")
    print("=" * 60)
    
    # Create agent that might produce invalid data
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(schema=ProductReview),
        system_prompt="Try to analyze the text, even if incomplete information is provided."
    )
    
    # Test with ambiguous input
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "This product is okay I guess. Not sure about rating."
        }]
    })
    
    # Check how validation worked
    review = result["structured_response"]
    print(f"🔍 Validation Handling:")
    print(f"   Rating: {review.rating} (None if unclear)")
    print(f"   Sentiment: {review.sentiment}")
    print(f"   Key Points: {review.key_points}")
    print()

if __name__ == "__main__":
    print("🌟 LangChain Structured Output Examples")
    print("This demo shows different ways to get structured data from LLMs\n")
    
    # Run all demos
    demo_automatic_schema()
    demo_tool_strategy()
    demo_custom_tool_message()
    demo_validation_handling()
    
    print("✅ All structured output demos completed!")
    print("💡 Structured output ensures reliable, parseable data from LLMs")