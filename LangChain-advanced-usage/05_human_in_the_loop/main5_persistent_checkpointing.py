"""
Persistent checkpointing for production Human-in-the-Loop workflows.

Shows how to:
- Use persistent checkpointers for production environments
- Handle long-running approval processes across sessions
- Implement conversation threading for multiple users
- Demonstrate state recovery and workflow resumption
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class ApprovalRecord:
    """Record of an approval request for persistence."""
    thread_id: str
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: float
    status: str = "pending"  # pending, approved, rejected, edited
    approver_id: Optional[str] = None
    feedback: Optional[str] = None

class PersistentApprovalManager:
    """Manages persistent approval records (simulates database)."""
    
    def __init__(self, storage_path: str = "/tmp/hitl_approvals.json"):
        self.storage_path = storage_path
        self._load_records()
    
    def _load_records(self):
        """Load approval records from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.records = {k: ApprovalRecord(**v) for k, v in data.items()}
            else:
                self.records = {}
        except Exception:
            self.records = {}
    
    def _save_records(self):
        """Save approval records to storage."""
        try:
            data = {k: v.__dict__ for k, v in self.records.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save approval records: {e}")
    
    def add_approval_request(self, record: ApprovalRecord) -> str:
        """Add a new approval request."""
        request_id = f"{record.thread_id}_{int(record.timestamp)}"
        self.records[request_id] = record
        self._save_records()
        return request_id
    
    def get_pending_approvals(self) -> Dict[str, ApprovalRecord]:
        """Get all pending approval requests."""
        return {k: v for k, v in self.records.items() if v.status == "pending"}
    
    def approve_request(self, request_id: str, approver_id: str, feedback: str = ""):
        """Approve an approval request."""
        if request_id in self.records:
            self.records[request_id].status = "approved"
            self.records[request_id].approver_id = approver_id
            self.records[request_id].feedback = feedback
            self._save_records()
    
    def reject_request(self, request_id: str, approver_id: str, feedback: str):
        """Reject an approval request."""
        if request_id in self.records:
            self.records[request_id].status = "rejected"
            self.records[request_id].approver_id = approver_id
            self.records[request_id].feedback = feedback
            self._save_records()

# Global approval manager for demo
approval_manager = PersistentApprovalManager()

@tool
def process_payment(amount: float, recipient: str, currency: str = "USD") -> str:
    """Process a payment transaction."""
    return f"Payment of {amount} {currency} processed to {recipient}"

@tool
def update_user_permissions(user_id: str, permissions: list[str]) -> str:
    """Update user permissions in the system."""
    return f"Updated permissions for user {user_id}: {', '.join(permissions)}"

@tool
def deploy_application(environment: str, version: str, rollback_enabled: bool = True) -> str:
    """Deploy application to specified environment."""
    rollback_note = "with rollback enabled" if rollback_enabled else "without rollback"
    return f"Deployed version {version} to {environment} {rollback_note}"

def create_hitl_agent(thread_id: str) -> Any:
    """Create a HITL agent with persistent checkpointing."""
    
    # In production, use AsyncPostgresSaver, RedisSaver, or DynamoDBSaver
    # For demo, using InMemorySaver but simulating persistence with files
    checkpointer = InMemorySaver()
    
    agent = create_agent(
        model,
        tools=[process_payment, update_user_permissions, deploy_application],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "process_payment": True,
                    "update_user_permissions": True,
                    "deploy_application": {
                        "allowed_decisions": ["approve", "reject"]  # No edits for deployments
                    }
                },
                description_prefix=f"Thread {thread_id} requires approval for"
            )
        ],
        checkpointer=checkpointer
    )
    
    return agent

def simulate_approval_workflow():
    """Simulate a complete approval workflow with persistence."""
    print("=== Persistent Checkpointing HITL Example ===\n")
    
    # Simulate multiple users/threads
    threads = [
        {"id": "finance_001", "user": "alice", "task": "Process payment of $5000 to vendor ABC Corp"},
        {"id": "admin_002", "user": "bob", "task": "Give admin permissions to user john_doe"},
        {"id": "deploy_003", "user": "charlie", "task": "Deploy version 2.1.0 to production environment"}
    ]
    
    print("🏗️ Step 1: Creating approval requests from multiple users")
    print("=" * 60)
    
    pending_agents = {}
    
    # Start workflows for each thread
    for thread in threads:
        thread_id = thread["id"]
        user = thread["user"]
        task = thread["task"]
        
        print(f"\n👤 User {user} (Thread: {thread_id})")
        print(f"Task: {task}")
        
        agent = create_hitl_agent(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Start the workflow
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config=config
        )
        
        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            action = interrupt_info["action_requests"][0]
            
            # Record the approval request
            approval_record = ApprovalRecord(
                thread_id=thread_id,
                tool_name=action["name"],
                arguments=action["arguments"],
                timestamp=time.time()
            )
            
            request_id = approval_manager.add_approval_request(approval_record)
            pending_agents[request_id] = (agent, config)
            
            print(f"🛑 Approval required for {action['name']}")
            print(f"   Request ID: {request_id}")
            print(f"   Status: Pending approval")
        else:
            print("✅ Task completed without approval")
    
    print(f"\n📋 Step 2: Review pending approvals")
    print("=" * 60)
    
    pending_approvals = approval_manager.get_pending_approvals()
    print(f"Total pending approvals: {len(pending_approvals)}")
    
    for request_id, record in pending_approvals.items():
        print(f"\n📝 Request {request_id}")
        print(f"   Thread: {record.thread_id}")
        print(f"   Tool: {record.tool_name}")
        print(f"   Arguments: {record.arguments}")
        print(f"   Timestamp: {time.ctime(record.timestamp)}")
    
    print(f"\n⚖️ Step 3: Process approvals (simulating human reviewers)")
    print("=" * 60)
    
    # Simulate human reviewers processing approvals
    approver_decisions = [
        ("supervisor_jane", "approve", "Payment approved for legitimate vendor"),
        ("security_admin", "reject", "Admin permissions rejected - user needs training first"),
        ("devops_lead", "approve", "Deployment approved for production release")
    ]
    
    request_ids = list(pending_approvals.keys())
    
    for i, (approver, decision, feedback) in enumerate(approver_decisions):
        if i < len(request_ids):
            request_id = request_ids[i]
            record = pending_approvals[request_id]
            
            print(f"\n👨‍💼 {approver} reviewing request {request_id}")
            print(f"   Tool: {record.tool_name}")
            print(f"   Decision: {decision.upper()}")
            print(f"   Feedback: {feedback}")
            
            if decision == "approve":
                approval_manager.approve_request(request_id, approver, feedback)
            else:
                approval_manager.reject_request(request_id, approver, feedback)
    
    print(f"\n🔄 Step 4: Resume workflows with approval decisions")
    print("=" * 60)
    
    # Resume each workflow with the approval decisions
    for request_id, record in pending_approvals.items():
        if request_id in pending_agents:
            agent, config = pending_agents[request_id]
            
            print(f"\n📤 Resuming workflow for thread {record.thread_id}")
            
            # Get the updated record with decision
            updated_record = approval_manager.records[request_id]
            
            if updated_record.status == "approved":
                decision = {"type": "approve"}
                print("✅ Continuing with approved action")
            else:
                decision = {"type": "reject", "feedback": updated_record.feedback}
                print(f"❌ Rejecting action: {updated_record.feedback}")
            
            # Resume the workflow
            try:
                resume_result = agent.invoke(
                    Command(resume={"decisions": [decision]}),
                    config=config
                )
                
                if "messages" in resume_result and resume_result["messages"]:
                    final_message = resume_result["messages"][-1]
                    print(f"   Result: {final_message.content}")
                else:
                    print("   Workflow completed")
                    
            except Exception as e:
                print(f"   Error resuming workflow: {e}")
    
    print(f"\n📊 Step 5: Final approval status")
    print("=" * 60)
    
    all_records = approval_manager.records
    status_counts = {"approved": 0, "rejected": 0, "pending": 0}
    
    for record in all_records.values():
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
        print(f"• {record.thread_id}: {record.tool_name} - {record.status.upper()}")
        if record.approver_id:
            print(f"  Reviewed by: {record.approver_id}")
        if record.feedback:
            print(f"  Feedback: {record.feedback}")
    
    print(f"\nSummary: {status_counts['approved']} approved, {status_counts['rejected']} rejected, {status_counts['pending']} pending")

def demonstrate_state_recovery():
    """Demonstrate state recovery after system restart."""
    print(f"\n" + "=" * 60)
    print("\n🔄 State Recovery Demonstration")
    print("Simulating system restart and workflow recovery...")
    
    # Check if there are any existing approval records
    existing_records = approval_manager.records
    
    if existing_records:
        print(f"\n📂 Found {len(existing_records)} existing approval records")
        
        for request_id, record in existing_records.items():
            print(f"• {request_id}: {record.tool_name} ({record.status})")
        
        print("\n✨ In a production system, these workflows could be:")
        print("- Automatically resumed from checkpoints")
        print("- Displayed in an approval dashboard")
        print("- Processed by different approvers")
        print("- Escalated if pending too long")
        
    else:
        print("\n📝 No existing records found (fresh start)")
    
    print("\n🔧 Cleanup: Removing demo approval records")
    if os.path.exists(approval_manager.storage_path):
        os.remove(approval_manager.storage_path)
        print("✅ Demo records cleaned up")

if __name__ == "__main__":
    simulate_approval_workflow()
    demonstrate_state_recovery()
    
    print("\n🎯 Key Production Features Demonstrated:")
    print("- Multi-threaded approval workflows")
    print("- Persistent approval record storage")
    print("- State recovery after system restart")
    print("- Approval tracking and audit trail")
    print("- Support for multiple concurrent users")
    print("- Workflow resumption from checkpoints")