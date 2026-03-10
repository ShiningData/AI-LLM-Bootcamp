"""
Interactive Human-in-the-Loop with real user input.

Shows how to:
- Create truly interactive approval workflows
- Wait for real human decisions via command line
- Handle user input for approve/edit/reject decisions
- Demonstrate actual human intervention in agent workflows
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv
import json

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
def transfer_money(amount: float, from_account: str, to_account: str) -> str:
    """Transfer money between accounts (requires approval)."""
    return f"Transferred ${amount} from {from_account} to {to_account}"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send email to a recipient (requires approval)."""
    return f"Email sent to {recipient} with subject '{subject}' (body: {len(body)} chars)"

def get_human_decision(action_request, allowed_decisions):
    """Get real human decision for an action request."""
    print("\n" + "="*60)
    print("🚨 HUMAN INTERVENTION REQUIRED!")
    print("="*60)
    print(f"Tool: {action_request['name']}")
    print(f"Description: {action_request['description']}")
    print("\nArguments:")
    for key, value in action_request['arguments'].items():
        print(f"  {key}: {value}")
    
    print(f"\nAllowed decisions: {', '.join(allowed_decisions)}")
    
    while True:
        print(f"\nWhat would you like to do?")
        for i, decision in enumerate(allowed_decisions, 1):
            print(f"  {i}. {decision.upper()}")
        
        try:
            choice = input("\nEnter your choice (number): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(allowed_decisions):
                decision_type = allowed_decisions[choice_idx]
                
                if decision_type == "edit":
                    return get_edit_decision(action_request)
                elif decision_type == "reject":
                    return get_reject_decision()
                else:  # approve
                    print("✅ Action approved!")
                    return {"type": "approve"}
            else:
                print("❌ Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Please enter a number.")

def get_edit_decision(action_request):
    """Get edited arguments from user."""
    print("\n🔧 EDITING MODE")
    print("Current arguments:")
    
    edited_args = action_request['arguments'].copy()
    
    for key, current_value in edited_args.items():
        print(f"\n{key} (current: {current_value})")
        new_value = input(f"New value (press Enter to keep current): ").strip()
        
        if new_value:
            # Try to preserve the original type
            if isinstance(current_value, (int, float)):
                try:
                    if isinstance(current_value, int):
                        edited_args[key] = int(new_value)
                    else:
                        edited_args[key] = float(new_value)
                except ValueError:
                    edited_args[key] = new_value  # Keep as string
            else:
                edited_args[key] = new_value
    
    print("\n✏️  Edited arguments:")
    for key, value in edited_args.items():
        print(f"  {key}: {value}")
    
    confirm = input("\nConfirm edits? (y/n): ").strip().lower()
    if confirm == 'y':
        print("✅ Arguments edited!")
        return {"type": "edit", "arguments": edited_args}
    else:
        print("❌ Edit cancelled, defaulting to reject")
        return {"type": "reject", "feedback": "User cancelled edit"}

def get_reject_decision():
    """Get rejection feedback from user."""
    feedback = input("\n❌ Rejection reason (optional): ").strip()
    print("🚫 Action rejected!")
    return {"type": "reject", "feedback": feedback or "Rejected by user"}

def interactive_hitl_demo():
    """Run interactive Human-in-the-Loop demonstration."""
    print("=== Interactive Human-in-the-Loop Example ===")
    print("This will ACTUALLY wait for your decisions!\n")
    
    # Create agent with HITL middleware
    agent = create_agent(
        model,
        tools=[read_user_data, delete_user_account, transfer_money, send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # Safe operation - no approval needed
                    "read_user_data": False,
                    
                    # DANGEROUS operation - full control
                    "delete_user_account": True,
                    
                    # Money transfer - full control
                    "transfer_money": True,
                    
                    # Email - approve/reject only (no editing)
                    "send_email": {
                        "allowed_decisions": ["approve", "reject"]
                    }
                },
                description_prefix="Human approval required for"
            )
        ],
        checkpointer=InMemorySaver()
    )
    
    config = {"configurable": {"thread_id": "interactive_demo"}}
    
    # Test scenarios
    scenarios = [
        {
            "name": "Safe Operation",
            "prompt": "Show me data for user 12345",
            "expected": "Should execute immediately without asking you"
        },
        {
            "name": "Dangerous Operation", 
            "prompt": "Delete user account 12345",
            "expected": "Should ask for your approval (recommend REJECT!)"
        },
        {
            "name": "Money Transfer",
            "prompt": "Transfer $1000 from account_A to account_B", 
            "expected": "Should ask for approval (you can edit the amount)"
        },
        {
            "name": "Email Sending",
            "prompt": "Send email to boss@company.com with subject 'Update' and body 'Project completed'",
            "expected": "Should ask for approval (approve/reject only)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        print('='*60)
        
        print(f"🤖 Agent Request: {scenario['prompt']}")
        
        # Start the workflow
        result = agent.invoke(
            {"messages": [{"role": "user", "content": scenario["prompt"]}]},
            config=config
        )
        
        # Handle interrupt if it occurs
        while "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            action_requests = interrupt_info["action_requests"]
            review_configs = interrupt_info["review_configs"]
            
            decisions = []
            
            # Process each action that needs approval
            for action, review_config in zip(action_requests, review_configs):
                allowed_decisions = review_config.get("allowed_decisions", ["approve", "edit", "reject"])
                decision = get_human_decision(action, allowed_decisions)
                decisions.append(decision)
            
            # Resume with human decisions
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config
            )
        
        # Show final result
        if "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            print(f"\n🎯 Final Result: {final_message.content}")
        
        input("\nPress Enter to continue to next scenario...")
    
    print(f"\n🎉 Interactive Human-in-the-Loop demonstration completed!")
    print("You experienced real human intervention in AI agent workflows!")

if __name__ == "__main__":
    try:
        interactive_hitl_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have GOOGLE_API_KEY set in your environment")