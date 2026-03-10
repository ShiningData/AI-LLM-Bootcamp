"""
Multi-step Human-in-the-Loop workflow with cascading approvals.

Shows how to:
- Handle multiple tool calls in a single agent response
- Implement cascading approval workflows
- Manage complex multi-step processes with human oversight
- Coordinate approvals across related operations
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@tool
def create_database_backup(database_name: str) -> str:
    """Create a backup of the specified database."""
    return f"Database backup created for '{database_name}' at backup_20241201_123456.sql"

@tool
def apply_schema_migration(database_name: str, migration_file: str) -> str:
    """Apply database schema migration."""
    return f"Migration '{migration_file}' applied successfully to database '{database_name}'"

@tool
def restart_application_service(service_name: str) -> str:
    """Restart an application service."""
    return f"Service '{service_name}' restarted successfully"

@tool
def update_load_balancer_config(config_changes: str) -> str:
    """Update load balancer configuration."""
    return f"Load balancer updated with changes: {config_changes}"

@tool
def send_deployment_notification(stakeholders: List[str], deployment_info: str) -> str:
    """Send deployment notification to stakeholders."""
    return f"Deployment notification sent to {len(stakeholders)} stakeholders: {deployment_info}"

def demonstrate_multi_step_approval():
    """Demonstrate multi-step Human-in-the-Loop approval workflow."""
    print("=== Multi-Step Human-in-the-Loop Approval Example ===\n")
    print("Simulating a complex deployment process with multiple approval points...")
    
    # Create agent with approval required for all deployment-related tools
    agent = create_agent(
        model,
        tools=[
            create_database_backup, 
            apply_schema_migration, 
            restart_application_service,
            update_load_balancer_config,
            send_deployment_notification
        ],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # All deployment operations require approval
                    "create_database_backup": True,
                    "apply_schema_migration": True,
                    "restart_application_service": True,
                    "update_load_balancer_config": True,
                    "send_deployment_notification": {
                        "allowed_decisions": ["approve", "reject"]  # No editing notifications
                    }
                },
                description_prefix="Deployment step requires approval"
            )
        ],
        checkpointer=InMemorySaver()
    )
    
    config = {"configurable": {"thread_id": "deployment_workflow"}}
    
    print("🚀 Deployment Request: Deploy new application version with database changes")
    print("This will trigger multiple approval points in sequence...\n")
    
    # Start the complex deployment process
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user", 
                    "content": "Deploy the new application version: First backup the main_db database, then apply migration_v2.1.sql, restart the app-server service, and notify the team"
                }
            ]
        },
        config=config
    )
    
    step_counter = 1
    
    # Process each approval step
    while "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value
        action_requests = interrupt_info["action_requests"]
        
        print(f"🛑 DEPLOYMENT STEP {step_counter} - Approval Required")
        print("=" * 50)
        
        decisions = []
        
        for i, action in enumerate(action_requests):
            tool_name = action["name"]
            arguments = action["arguments"]
            
            print(f"\n📋 Action {i+1}: {tool_name}")
            print(f"Arguments: {arguments}")
            
            # Simulate human decision-making for each step
            if tool_name == "create_database_backup":
                print("💭 Human Review: Database backup is safe and necessary")
                print("✅ Decision: APPROVE")
                decisions.append({"type": "approve"})
                
            elif tool_name == "apply_schema_migration":
                print("💭 Human Review: Migration file needs verification")
                
                # Check if this is a potentially risky migration
                migration_file = arguments.get("migration_file", "")
                if "DROP" in migration_file.upper() or "DELETE" in migration_file.upper():
                    print("🚨 WARNING: Migration contains destructive operations")
                    print("❌ Decision: REJECT")
                    decisions.append({
                        "type": "reject", 
                        "feedback": "Migration rejected - contains destructive operations. Please review and resubmit."
                    })
                else:
                    print("✅ Decision: APPROVE")
                    decisions.append({"type": "approve"})
                    
            elif tool_name == "restart_application_service":
                print("💭 Human Review: Service restart during business hours")
                
                # Simulate checking business hours
                import datetime
                current_hour = datetime.datetime.now().hour
                
                if 9 <= current_hour <= 17:  # Business hours
                    print("⚠️  WARNING: Restart requested during business hours")
                    print("🔧 Decision: EDIT (schedule for maintenance window)")
                    
                    # Edit to add a delay or different service
                    edited_args = arguments.copy()
                    edited_args["service_name"] = f"staging-{arguments['service_name']}"
                    decisions.append({
                        "type": "edit",
                        "arguments": edited_args
                    })
                else:
                    print("✅ Decision: APPROVE (outside business hours)")
                    decisions.append({"type": "approve"})
                    
            elif tool_name == "update_load_balancer_config":
                print("💭 Human Review: Load balancer configuration change")
                print("✅ Decision: APPROVE (configuration looks safe)")
                decisions.append({"type": "approve"})
                
            elif tool_name == "send_deployment_notification":
                print("💭 Human Review: Deployment notification")
                print("✅ Decision: APPROVE (team should be notified)")
                decisions.append({"type": "approve"})
            
            else:
                print("❓ Unknown tool - defaulting to APPROVE")
                decisions.append({"type": "approve"})
        
        print(f"\n📝 Submitting {len(decisions)} decisions for step {step_counter}")
        
        # Resume execution with the collected decisions
        result = agent.invoke(
            Command(
                resume={
                    "decisions": decisions
                }
            ),
            config=config
        )
        
        step_counter += 1
        print()
    
    # Show final result
    print("🎉 DEPLOYMENT COMPLETED!")
    print("=" * 50)
    if "messages" in result and result["messages"]:
        final_message = result["messages"][-1]
        print(f"Final Result: {final_message.content}")
    
    print(f"\nDeployment workflow completed with {step_counter - 1} approval steps.")
    
    print("\n📊 Approval Summary:")
    print("✅ Database backup: Approved")
    print("❌ Schema migration: Rejected (contained destructive operations)")
    print("🔧 Service restart: Edited (moved to staging)")
    print("✅ Load balancer update: Approved")
    print("✅ Team notification: Approved")

def demonstrate_batch_approval():
    """Demonstrate handling multiple simultaneous approvals."""
    print("\n" + "=" * 60)
    print("\n=== Batch Approval Example ===")
    print("Handling multiple related operations that need approval simultaneously...")
    
    agent = create_agent(
        model,
        tools=[create_database_backup, restart_application_service, send_deployment_notification],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on=True,  # All tools require approval
                description_prefix="Batch operation requires approval"
            )
        ],
        checkpointer=InMemorySaver()
    )
    
    config = {"configurable": {"thread_id": "batch_approval_demo"}}
    
    # Request multiple related operations
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Perform emergency maintenance: backup all databases, restart critical services, and notify the on-call team"
                }
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value
        action_requests = interrupt_info["action_requests"]
        
        print(f"🛑 BATCH APPROVAL REQUIRED")
        print(f"Number of operations: {len(action_requests)}")
        
        # Show all operations that need approval
        for i, action in enumerate(action_requests):
            print(f"\n{i+1}. {action['name']}")
            print(f"   Args: {action['arguments']}")
        
        # Simulate batch decision
        print("\n💭 Human Review: Emergency maintenance operations")
        print("🚨 All operations approved for emergency maintenance")
        
        # Approve all operations in batch
        decisions = [{"type": "approve"} for _ in action_requests]
        
        resume_result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )
        
        print("\n✅ All emergency operations completed:")
        if "messages" in resume_result and resume_result["messages"]:
            final_message = resume_result["messages"][-1]
            print(f"Result: {final_message.content}")

if __name__ == "__main__":
    demonstrate_multi_step_approval()
    demonstrate_batch_approval()
    
    print("\n🎯 Key Takeaways:")
    print("- Complex workflows can have multiple approval points")
    print("- Each step can have different approval policies")
    print("- Decisions can be approve/edit/reject based on context")
    print("- Batch operations can be approved simultaneously")
    print("- State is preserved between approval steps")