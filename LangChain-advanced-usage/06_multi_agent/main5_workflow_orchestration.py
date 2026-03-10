"""
Workflow Orchestration Multi-Agent System.

Shows how to:
- Create workflow orchestrator that manages complex multi-step processes
- Implement sequential and parallel task execution
- Handle task dependencies and error recovery
- Coordinate multiple agents for business process automation
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, List

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

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

# Workflow management tools
@tool
def execute_task(task_id: str, task_description: str, dependencies: str = "") -> str:
    """Execute a specific task in the workflow."""
    return f"TASK_EXECUTION: {task_id}\nDescription: {task_description}\nDependencies: {dependencies}\nStatus: Task executed successfully"

@tool
def check_dependencies(task_id: str, required_tasks: str) -> str:
    """Check if all dependencies for a task are satisfied."""
    return f"DEPENDENCY_CHECK: {task_id}\nRequired: {required_tasks}\nStatus: Dependencies verified and satisfied"

@tool
def update_workflow_status(workflow_id: str, task_id: str, status: str) -> str:
    """Update the status of a task in the workflow."""
    return f"STATUS_UPDATE: Workflow {workflow_id}, Task {task_id} set to {status}\nTimestamp: Updated successfully"

@tool
def handle_task_failure(task_id: str, error_reason: str, recovery_action: str) -> str:
    """Handle task failure with appropriate recovery action."""
    return f"ERROR_HANDLING: {task_id}\nReason: {error_reason}\nRecovery: {recovery_action}\nStatus: Recovery action initiated"

@tool
def notify_stakeholders(workflow_id: str, message: str, recipients: str) -> str:
    """Send notifications to workflow stakeholders."""
    return f"NOTIFICATION: Workflow {workflow_id}\nMessage: {message}\nRecipients: {recipients}\nStatus: Notifications sent"

class WorkflowOrchestrator:
    """Manages complex multi-agent workflows with task dependencies."""
    
    def __init__(self):
        self.workflows = {}
        self.task_status = {}
        
        # Create specialized workflow agents
        self.agents = {
            "coordinator": self._create_coordinator_agent(),
            "validator": self._create_validator_agent(),
            "executor": self._create_executor_agent(),
            "monitor": self._create_monitor_agent()
        }
    
    def _create_coordinator_agent(self):
        """Create workflow coordination agent."""
        return create_agent(
            model,
            tools=[check_dependencies, update_workflow_status, notify_stakeholders],
            system_prompt="""You are a workflow coordinator responsible for managing task execution order and dependencies.

Your responsibilities:
- Use check_dependencies to verify prerequisites before task execution
- Use update_workflow_status to track progress through workflows
- Use notify_stakeholders to communicate important workflow events
- Ensure tasks execute in the correct sequence
- Coordinate between different workflow agents

Focus on maintaining workflow integrity and proper sequencing."""
        )
    
    def _create_validator_agent(self):
        """Create task validation agent."""
        return create_agent(
            model,
            tools=[execute_task, update_workflow_status],
            system_prompt="""You are a task validator responsible for ensuring task quality and completeness.

Your responsibilities:
- Use execute_task for validation and quality assurance tasks
- Use update_workflow_status to mark validation results
- Verify task outputs meet quality standards
- Check for completeness and accuracy
- Approve or reject task results

Focus on quality control and validation of work products."""
        )
    
    def _create_executor_agent(self):
        """Create task execution agent."""
        return create_agent(
            model,
            tools=[execute_task, update_workflow_status, handle_task_failure],
            system_prompt="""You are a task executor responsible for performing the actual work in workflows.

Your responsibilities:
- Use execute_task to perform assigned work tasks
- Use update_workflow_status to report progress and completion
- Use handle_task_failure to manage errors and exceptions
- Execute tasks efficiently and accurately
- Report issues promptly

Focus on reliable task execution and error handling."""
        )
    
    def _create_monitor_agent(self):
        """Create workflow monitoring agent."""
        return create_agent(
            model,
            tools=[update_workflow_status, handle_task_failure, notify_stakeholders],
            system_prompt="""You are a workflow monitor responsible for tracking progress and handling issues.

Your responsibilities:
- Use update_workflow_status to track overall workflow health
- Use handle_task_failure to respond to problems and bottlenecks
- Use notify_stakeholders to communicate status and issues
- Monitor workflow performance and identify problems
- Escalate issues when necessary

Focus on proactive monitoring and issue resolution."""
        )
    
    def execute_workflow(self, workflow_definition: Dict) -> Dict:
        """Execute a complete workflow with multiple coordinated agents."""
        workflow_id = workflow_definition['id']
        tasks = workflow_definition['tasks']
        
        print(f"🔄 EXECUTING WORKFLOW: {workflow_id}")
        print("=" * 50)
        
        results = {
            'workflow_id': workflow_id,
            'task_results': {},
            'execution_log': []
        }
        
        # Phase 1: Workflow Planning and Coordination
        print("\n--- PHASE 1: WORKFLOW COORDINATION ---")
        coordinator = self.agents['coordinator']
        
        coordination_result = coordinator.invoke({
            "messages": [{"role": "user", "content": f"Coordinate the execution of workflow '{workflow_id}' with these tasks: {[t['id'] for t in tasks]}. Check dependencies and plan execution order."}]
        })
        
        coordination_output = extract_clean_content(coordination_result['messages'][-1].content)
        results['execution_log'].append(f"Coordination: {coordination_output}")
        print(f"Coordination complete: {coordination_output[:150]}...")
        
        # Phase 2: Task Execution
        print("\n--- PHASE 2: TASK EXECUTION ---")
        executor = self.agents['executor']
        
        for task in tasks:
            task_id = task['id']
            task_desc = task['description']
            dependencies = task.get('dependencies', [])
            
            print(f"\nExecuting task: {task_id}")
            
            # Execute task
            execution_result = executor.invoke({
                "messages": [{"role": "user", "content": f"Execute task '{task_id}': {task_desc}. Dependencies: {dependencies}"}]
            })
            
            task_output = extract_clean_content(execution_result['messages'][-1].content)
            results['task_results'][task_id] = task_output
            results['execution_log'].append(f"Task {task_id}: {task_output}")
            print(f"Task {task_id} completed: {task_output[:100]}...")
        
        # Phase 3: Validation
        print("\n--- PHASE 3: WORKFLOW VALIDATION ---")
        validator = self.agents['validator']
        
        validation_result = validator.invoke({
            "messages": [{"role": "user", "content": f"Validate the completion of workflow '{workflow_id}'. All tasks should be completed successfully: {list(results['task_results'].keys())}"}]
        })
        
        validation_output = extract_clean_content(validation_result['messages'][-1].content)
        results['execution_log'].append(f"Validation: {validation_output}")
        print(f"Validation complete: {validation_output[:150]}...")
        
        # Phase 4: Monitoring and Reporting
        print("\n--- PHASE 4: WORKFLOW MONITORING ---")
        monitor = self.agents['monitor']
        
        monitoring_result = monitor.invoke({
            "messages": [{"role": "user", "content": f"Provide final monitoring report for workflow '{workflow_id}'. Summarize execution status and notify stakeholders of completion."}]
        })
        
        monitoring_output = extract_clean_content(monitoring_result['messages'][-1].content)
        results['execution_log'].append(f"Monitoring: {monitoring_output}")
        print(f"Monitoring complete: {monitoring_output[:150]}...")
        
        return results

def demonstrate_simple_workflow():
    """Demonstrate a simple sequential workflow."""
    print("=== Simple Sequential Workflow ===\n")
    
    orchestrator = WorkflowOrchestrator()
    
    # Define a simple workflow
    workflow = {
        'id': 'user_onboarding',
        'tasks': [
            {'id': 'create_account', 'description': 'Create new user account with credentials', 'dependencies': []},
            {'id': 'send_welcome_email', 'description': 'Send welcome email to new user', 'dependencies': ['create_account']},
            {'id': 'setup_profile', 'description': 'Initialize user profile with default settings', 'dependencies': ['create_account']},
            {'id': 'assign_permissions', 'description': 'Assign appropriate user permissions', 'dependencies': ['setup_profile']}
        ]
    }
    
    results = orchestrator.execute_workflow(workflow)
    
    print("\n" + "="*50)
    print("WORKFLOW EXECUTION SUMMARY:")
    print(f"Workflow ID: {results['workflow_id']}")
    print(f"Tasks completed: {len(results['task_results'])}")
    print(f"Execution steps: {len(results['execution_log'])}")

def demonstrate_complex_workflow():
    """Demonstrate a complex workflow with parallel tasks."""
    print("\n=== Complex Parallel Workflow ===\n")
    
    orchestrator = WorkflowOrchestrator()
    
    # Define a complex workflow
    workflow = {
        'id': 'product_launch',
        'tasks': [
            {'id': 'market_research', 'description': 'Conduct market research and competitive analysis', 'dependencies': []},
            {'id': 'product_development', 'description': 'Develop product features and functionality', 'dependencies': []},
            {'id': 'marketing_materials', 'description': 'Create marketing content and materials', 'dependencies': ['market_research']},
            {'id': 'quality_testing', 'description': 'Perform comprehensive product testing', 'dependencies': ['product_development']},
            {'id': 'launch_coordination', 'description': 'Coordinate product launch activities', 'dependencies': ['marketing_materials', 'quality_testing']}
        ]
    }
    
    results = orchestrator.execute_workflow(workflow)
    
    print("\n" + "="*50)
    print("COMPLEX WORKFLOW SUMMARY:")
    print(f"Workflow ID: {results['workflow_id']}")
    print(f"Parallel tasks handled: {len([t for t in workflow['tasks'] if not t['dependencies']])}")
    print(f"Dependencies managed: {sum(len(t.get('dependencies', [])) for t in workflow['tasks'])}")

def demonstrate_error_handling():
    """Demonstrate workflow error handling and recovery."""
    print("\n=== Error Handling Workflow ===\n")
    
    orchestrator = WorkflowOrchestrator()
    
    # Simulate workflow with potential failure points
    workflow = {
        'id': 'data_processing',
        'tasks': [
            {'id': 'data_extraction', 'description': 'Extract data from external sources', 'dependencies': []},
            {'id': 'data_validation', 'description': 'Validate extracted data quality', 'dependencies': ['data_extraction']},
            {'id': 'data_transformation', 'description': 'Transform data to required format', 'dependencies': ['data_validation']},
            {'id': 'error_recovery', 'description': 'Handle any data processing errors', 'dependencies': []}
        ]
    }
    
    results = orchestrator.execute_workflow(workflow)
    
    print("\n" + "="*50)
    print("ERROR HANDLING SUMMARY:")
    print(f"Error handling tasks: {len([t for t in workflow['tasks'] if 'error' in t['description']])}")
    print(f"Recovery mechanisms: Integrated in workflow execution")

if __name__ == "__main__":
    print("🔄 Workflow Orchestration Multi-Agent System")
    print("This system coordinates multiple agents to execute complex business workflows\n")
    
    demonstrate_simple_workflow()
    demonstrate_complex_workflow()
    demonstrate_error_handling()
    
    print("\n✅ Workflow Orchestration Benefits:")
    print("   🎯 Complex process automation")
    print("   🔄 Sequential and parallel task coordination")
    print("   📋 Dependency management") 
    print("   🛡️ Error handling and recovery")
    print("   📊 Progress monitoring and reporting")