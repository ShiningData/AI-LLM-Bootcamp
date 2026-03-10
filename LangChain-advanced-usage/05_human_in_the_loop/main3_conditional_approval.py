"""
Conditional Human-in-the-Loop approval based on argument values.

Shows how to:
- Implement conditional approval policies based on tool arguments
- Use custom logic to determine when human oversight is needed
- Handle different risk levels within the same tool
- Create dynamic approval rules
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class ApprovalContext:
    """Context for making approval decisions."""
    tool_name: str
    arguments: Dict[str, Any]
    user_role: str = "standard_user"

def requires_approval(context: ApprovalContext) -> bool:
    """Custom logic to determine if approval is needed."""
    tool_name = context.tool_name
    args = context.arguments
    
    if tool_name == "transfer_funds":
        # Large transfers always require approval
        amount = args.get("amount", 0)
        if amount > 1000:
            return True
        # Transfers to external accounts require approval
        if args.get("to_account", "").startswith("external_"):
            return True
        return False
    
    elif tool_name == "modify_system_config":
        # Production configs always require approval
        environment = args.get("environment", "")
        if environment == "production":
            return True
        # Critical settings require approval
        setting_name = args.get("setting_name", "")
        critical_settings = ["database_url", "api_keys", "security_policy"]
        if setting_name in critical_settings:
            return True
        return False
    
    elif tool_name == "send_notification":
        # Mass notifications require approval
        recipient_count = len(args.get("recipients", []))
        if recipient_count > 10:
            return True
        # Emergency notifications always require approval
        if args.get("priority", "").lower() == "emergency":
            return True
        return False
    
    return False

@tool
def transfer_funds(amount: float, from_account: str, to_account: str, memo: str = "") -> str:
    """Transfer funds between accounts."""
    return f"Transferred ${amount} from {from_account} to {to_account}. Memo: {memo}"

@tool
def modify_system_config(environment: str, setting_name: str, setting_value: str) -> str:
    """Modify system configuration settings."""
    return f"Updated {setting_name} = {setting_value} in {environment} environment"

@tool
def send_notification(recipients: list[str], message: str, priority: str = "normal") -> str:
    """Send notification to recipients."""
    return f"Sent {priority} notification to {len(recipients)} recipients: {message}"

class ConditionalHITLMiddleware(HumanInTheLoopMiddleware):
    """Custom HITL middleware with conditional approval logic."""
    
    def __init__(self, approval_function, **kwargs):
        self.approval_function = approval_function
        # Start with all tools requiring approval, we'll filter in the hook
        super().__init__(interrupt_on=True, **kwargs)
    
    def should_interrupt(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Override to implement custom approval logic."""
        context = ApprovalContext(tool_name=tool_name, arguments=arguments)
        return self.approval_function(context)

def demonstrate_conditional_approval():
    """Demonstrate conditional Human-in-the-Loop approval."""
    print("=== Conditional Human-in-the-Loop Approval Example ===\n")
    
    # Create agent with conditional approval middleware
    agent = create_agent(
        model,
        tools=[transfer_funds, modify_system_config, send_notification],
        middleware=[
            # Note: Using standard middleware for this demo
            # In practice, you'd use the custom ConditionalHITLMiddleware above
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # We'll simulate conditional approval by testing different scenarios
                    "transfer_funds": True,
                    "modify_system_config": True, 
                    "send_notification": True
                },
                description_prefix="Conditional approval required for"
            )
        ],
        checkpointer=InMemorySaver()
    )
    
    config = {"configurable": {"thread_id": "conditional_demo"}}
    
    print("💰 Test 1: Small transfer (should be low risk)")
    print("Transfer $50 from internal account...\n")
    
    result1 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Transfer $50 from account_123 to account_456 with memo 'coffee money'"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result1:
        interrupt_info = result1["__interrupt__"][0].value
        action = interrupt_info["action_requests"][0]
        
        # Simulate conditional approval logic
        amount = action["arguments"]["amount"]
        to_account = action["arguments"]["to_account"]
        
        print("🛑 Transfer interrupted for review:")
        print(f"  Amount: ${amount}")
        print(f"  To account: {to_account}")
        
        # Small transfer - auto approve
        if amount <= 1000 and not to_account.startswith("external_"):
            print("✅ AUTO-APPROVED: Small internal transfer")
            decision = {"type": "approve"}
        else:
            print("❌ REQUIRES MANUAL REVIEW: High risk transfer")
            decision = {"type": "reject", "feedback": "High risk transfer requires manual review"}
        
        resume_result = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config=config
        )
        
        final_message = resume_result["messages"][-1]
        print(f"Result: {final_message.content}")
    
    print("\n" + "=" * 60)
    print("\n💰 Test 2: Large transfer (should require approval)")
    
    result2 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Transfer $5000 from account_123 to external_bank_999"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result2:
        interrupt_info = result2["__interrupt__"][0].value
        action = interrupt_info["action_requests"][0]
        
        amount = action["arguments"]["amount"]
        to_account = action["arguments"]["to_account"]
        
        print("🛑 Large transfer interrupted:")
        print(f"  Amount: ${amount}")
        print(f"  To account: {to_account}")
        print("🚨 HIGH RISK: Large amount + external account")
        print("❌ Human decision: REJECT")
        
        resume_result = agent.invoke(
            Command(resume={
                "decisions": [{
                    "type": "reject",
                    "feedback": "Large external transfer rejected - requires additional verification"
                }]
            }),
            config=config
        )
        
        final_message = resume_result["messages"][-1]
        print(f"Result: {final_message.content}")
    
    print("\n" + "=" * 60)
    print("\n⚙️ Test 3: System configuration change")
    
    result3 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Update database_url setting to 'new_db_connection' in production environment"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result3:
        interrupt_info = result3["__interrupt__"][0].value
        action = interrupt_info["action_requests"][0]
        
        environment = action["arguments"]["environment"]
        setting_name = action["arguments"]["setting_name"]
        
        print("🛑 System config change interrupted:")
        print(f"  Environment: {environment}")
        print(f"  Setting: {setting_name}")
        
        # Production + critical setting = high risk
        if environment == "production" and setting_name in ["database_url", "api_keys", "security_policy"]:
            print("🚨 CRITICAL: Production database configuration change")
            print("🔧 Human decision: EDIT (change to staging first)")
            
            edited_args = action["arguments"].copy()
            edited_args["environment"] = "staging"
            
            decision = {
                "type": "edit", 
                "arguments": edited_args
            }
        else:
            decision = {"type": "approve"}
        
        resume_result = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config=config
        )
        
        final_message = resume_result["messages"][-1]
        print(f"Result: {final_message.content}")
    
    print("\n" + "=" * 60)
    print("\n📢 Test 4: Notification sending")
    
    result4 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Send emergency notification 'System maintenance in 5 minutes' to all users"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result4:
        interrupt_info = result4["__interrupt__"][0].value
        action = interrupt_info["action_requests"][0]
        
        message = action["arguments"]["message"]
        priority = action["arguments"].get("priority", "normal")
        
        print("🛑 Notification sending interrupted:")
        print(f"  Message: {message}")
        print(f"  Priority: {priority}")
        
        # Emergency notifications require approval
        if priority == "emergency":
            print("🚨 EMERGENCY notification requires approval")
            print("✅ Human decision: APPROVE (legitimate maintenance notice)")
            decision = {"type": "approve"}
        else:
            decision = {"type": "approve"}
        
        resume_result = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config=config
        )
        
        final_message = resume_result["messages"][-1]
        print(f"Result: {final_message.content}")
    
    print("\n🎉 Conditional approval demonstration completed!")
    print("\nConditional approval rules demonstrated:")
    print("- Transfers: Amount > $1000 OR external accounts")
    print("- Config: Production environment OR critical settings")
    print("- Notifications: >10 recipients OR emergency priority")

if __name__ == "__main__":
    demonstrate_conditional_approval()