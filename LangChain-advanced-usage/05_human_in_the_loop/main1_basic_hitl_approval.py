"""
Basic Human-in-the-Loop approval for dangerous tools.

Shows how to:
- Configure HumanInTheLoopMiddleware with per-tool policies
- Set up checkpointing for interrupt/resume workflows
- Handle simple approve/reject decisions
- Pause execution before dangerous tool calls
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@tool
def read_user_data(user_id: str) -> str:
    """Read user data from the database (safe operation)."""
    return f"User data for {user_id}: Name=John Doe, Email=john@example.com, Status=Active"

@tool
def delete_user_account(user_id: str) -> str:
    """Delete a user account permanently (DANGEROUS operation)."""
    return f"WARNING: User account {user_id} has been permanently deleted!"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send email to a recipient (requires approval)."""
    return f"Email sent to {recipient} with subject '{subject}' (body: {len(body)} chars)"

def demonstrate_basic_hitl():
    """Demonstrate basic Human-in-the-Loop approval workflow."""
    print("=== Basic Human-in-the-Loop Approval Example ===\n")
    
    # Create agent with Human-in-the-Loop middleware
    agent = create_agent(
        model,
        tools=[read_user_data, delete_user_account, send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # Safe operation - no approval needed
                    "read_user_data": False,
                    
                    # DANGEROUS operation - always requires approval
                    "delete_user_account": True,
                    
                    # Potentially sensitive - requires approval but no edits
                    "send_email": {
                        "allowed_decisions": ["approve", "reject"]
                    }
                },
                description_prefix="Tool execution requires human approval"
            )
        ],
        checkpointer=InMemorySaver()  # Required for interrupt handling
    )
    
    # Configuration with thread ID (required for checkpointing)
    config = {"configurable": {"thread_id": "demo_conversation_1"}}
    
    print("🔍 Step 1: Testing safe operation (read_user_data)")
    print("This should execute immediately without interruption...\n")
    
    # Test safe operation - should execute without interruption
    result1 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Show me the data for user ID 12345"}
            ]
        },
        config=config
    )
    
    print("✅ Safe operation completed:")
    final_message = result1["messages"][-1]
    print(f"Result: {final_message.content}\n")
    print("=" * 60)
    
    print("\n🚨 Step 2: Testing dangerous operation (delete_user_account)")
    print("This should be interrupted and require human approval...\n")
    
    # Test dangerous operation - should be interrupted
    result2 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Delete user account for user ID 12345"}
            ]
        },
        config=config
    )
    
    # Check if there was an interrupt
    if "__interrupt__" in result2:
        interrupt_info = result2["__interrupt__"][0].value
        print("🛑 INTERRUPT DETECTED!")
        print("Action requests:")
        for action in interrupt_info["action_requests"]:
            print(f"  - Tool: {action['name']}")
            print(f"  - Arguments: {action['arguments']}")
            print(f"  - Description: {action['description']}")
        
        print("\nReview configurations:")
        for config_info in interrupt_info["review_configs"]:
            print(f"  - Tool: {config_info['action_name']}")
            print(f"  - Allowed decisions: {config_info['allowed_decisions']}")
        
        print("\n" + "=" * 60)
        print("\n🤔 Step 3: Human decision simulation")
        
        # Simulate human decision - REJECT the dangerous operation
        print("Human decision: REJECT (too dangerous!)")
        
        resume_result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "reject", "feedback": "Account deletion rejected for safety"}
                    ]
                }
            ),
            config=config
        )
        
        print("\n✅ Execution resumed with REJECT decision:")
        final_message = resume_result["messages"][-1]
        print(f"Result: {final_message.content}")
        
    else:
        print("❌ Expected interrupt did not occur!")
    
    print("\n" + "=" * 60)
    print("\n📧 Step 4: Testing email operation with APPROVAL")
    
    # Test email operation - should be interrupted but then approved
    email_result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Send an email to admin@company.com with subject 'Daily Report' and body 'Report attached'"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in email_result:
        print("🛑 Email operation interrupted for approval")
        
        # Simulate human decision - APPROVE the email
        print("Human decision: APPROVE (email is safe to send)")
        
        email_resume = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            ),
            config=config
        )
        
        print("\n✅ Email sent after human approval:")
        final_message = email_resume["messages"][-1]
        print(f"Result: {final_message.content}")
        
    print("\n🎉 Human-in-the-Loop demonstration completed!")

if __name__ == "__main__":
    demonstrate_basic_hitl()