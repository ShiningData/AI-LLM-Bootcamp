"""
Hierarchical Multi-Agent System.

Shows how to:
- Create hierarchical agent structures with managers and subordinates
- Implement delegation and escalation patterns
- Handle complex task breakdown and assignment
- Coordinate agents at different organizational levels
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
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
    max_tokens=400
)

# Hierarchical coordination tools
@tool
def delegate_task(subordinate_agent: str, task_description: str, priority: str = "normal") -> str:
    """Delegate a task to a subordinate agent."""
    return f"DELEGATION: Task assigned to {subordinate_agent}\nTask: {task_description}\nPriority: {priority}\nStatus: Task delegated successfully"

@tool
def escalate_issue(manager_agent: str, issue_description: str, urgency: str = "normal") -> str:
    """Escalate an issue to a higher-level manager."""
    return f"ESCALATION: Issue escalated to {manager_agent}\nIssue: {issue_description}\nUrgency: {urgency}\nStatus: Escalation initiated"

@tool
def report_progress(manager_agent: str, task_status: str, completion_percentage: str) -> str:
    """Report task progress to manager."""
    return f"PROGRESS_REPORT: To {manager_agent}\nStatus: {task_status}\nCompletion: {completion_percentage}%\nReported: Progress update sent"

@tool
def approve_decision(decision_description: str, approval_level: str) -> str:
    """Approve a decision at the appropriate organizational level."""
    return f"APPROVAL: Decision approved at {approval_level} level\nDecision: {decision_description}\nStatus: Approval granted"

@tool
def coordinate_teams(team_list: str, coordination_task: str) -> str:
    """Coordinate activities between multiple teams."""
    return f"TEAM_COORDINATION: Managing {team_list}\nTask: {coordination_task}\nStatus: Cross-team coordination active"

class HierarchicalAgentSystem:
    """Manages hierarchical agent structures with delegation and escalation."""
    
    def __init__(self):
        self.agent_hierarchy = {}
        self.task_assignments = {}
        
    def create_executive_agent(self, role: str):
        """Create executive-level agent for strategic decisions."""
        return create_agent(
            model,
            tools=[delegate_task, approve_decision, coordinate_teams],
            system_prompt=f"""You are a {role} focused on strategic oversight and high-level coordination.

Your responsibilities:
- Use delegate_task to assign major initiatives to department managers
- Use approve_decision for strategic decisions requiring executive approval
- Use coordinate_teams to manage cross-departmental initiatives
- Focus on strategic planning and organizational alignment
- Make high-level resource allocation decisions

Your level: Executive
Authority: Strategic decisions, resource allocation, cross-departmental coordination
Reports to: Board/CEO (external)"""
        )
    
    def create_manager_agent(self, department: str):
        """Create manager-level agent for departmental coordination."""
        return create_agent(
            model,
            tools=[delegate_task, escalate_issue, report_progress, approve_decision],
            system_prompt=f"""You are a {department} manager responsible for departmental operations and team coordination.

Your responsibilities:
- Use delegate_task to assign work to team leads and specialists
- Use escalate_issue to raise strategic decisions to executives
- Use report_progress to update executives on departmental status
- Use approve_decision for departmental-level approvals
- Coordinate team activities and resource management

Your department: {department}
Authority: Departmental decisions, team coordination, budget management
Reports to: Executive level"""
        )
    
    def create_team_lead_agent(self, specialization: str):
        """Create team lead agent for specialized team management."""
        return create_agent(
            model,
            tools=[delegate_task, escalate_issue, report_progress],
            system_prompt=f"""You are a team lead for {specialization} responsible for direct team management and task execution.

Your responsibilities:
- Use delegate_task to assign specific tasks to team members
- Use escalate_issue to raise complex problems to department managers
- Use report_progress to update managers on team status
- Provide technical guidance and task coordination
- Ensure quality and timely delivery of team outputs

Your specialization: {specialization}
Authority: Task assignment, quality control, team guidance
Reports to: Department manager"""
        )
    
    def create_specialist_agent(self, expertise: str):
        """Create specialist agent for individual contributor work."""
        return create_agent(
            model,
            tools=[escalate_issue, report_progress],
            system_prompt=f"""You are a {expertise} specialist responsible for executing specific tasks and providing expert input.

Your responsibilities:
- Use escalate_issue to raise technical problems to team leads
- Use report_progress to update team leads on task status
- Execute assigned tasks with high quality and expertise
- Provide technical recommendations and insights
- Focus on detailed implementation and problem-solving

Your expertise: {expertise}
Authority: Task execution, technical recommendations
Reports to: Team lead"""
        )

def demonstrate_hierarchical_task_flow():
    """Demonstrate how tasks flow through the hierarchy."""
    print("=== Hierarchical Task Flow ===\n")
    
    system = HierarchicalAgentSystem()
    
    # Create hierarchical structure
    ceo = system.create_executive_agent("CEO")
    engineering_manager = system.create_manager_agent("Engineering")
    ai_team_lead = system.create_team_lead_agent("AI Development")
    ml_specialist = system.create_specialist_agent("Machine Learning")
    
    # Simulate task flow from top to bottom
    print("--- EXECUTIVE LEVEL: Strategic Initiative ---")
    executive_result = ceo.invoke({
        "messages": [{"role": "user", "content": "We need to implement AI capabilities across our platform. Plan the strategic approach and delegate implementation."}]
    })
    executive_decision = extract_clean_content(executive_result['messages'][-1].content)
    print(f"CEO Decision: {executive_decision[:200]}...\n")
    
    print("--- MANAGEMENT LEVEL: Departmental Planning ---")
    manager_result = engineering_manager.invoke({
        "messages": [{"role": "user", "content": "CEO has approved AI implementation initiative. Break down into engineering tasks and assign to appropriate teams."}]
    })
    manager_plan = extract_clean_content(manager_result['messages'][-1].content)
    print(f"Engineering Manager Plan: {manager_plan[:200]}...\n")
    
    print("--- TEAM LEAD LEVEL: Team Coordination ---")
    lead_result = ai_team_lead.invoke({
        "messages": [{"role": "user", "content": "Engineering manager assigned AI implementation to our team. Coordinate team efforts and assign specific ML tasks."}]
    })
    lead_coordination = extract_clean_content(lead_result['messages'][-1].content)
    print(f"AI Team Lead Coordination: {lead_coordination[:200]}...\n")
    
    print("--- SPECIALIST LEVEL: Task Execution ---")
    specialist_result = ml_specialist.invoke({
        "messages": [{"role": "user", "content": "Team lead assigned ML model development task. Execute the implementation and report progress."}]
    })
    specialist_work = extract_clean_content(specialist_result['messages'][-1].content)
    print(f"ML Specialist Execution: {specialist_work[:200]}...\n")

def demonstrate_escalation_flow():
    """Demonstrate how issues escalate up the hierarchy."""
    print("=== Issue Escalation Flow ===\n")
    
    system = HierarchicalAgentSystem()
    
    # Create hierarchy
    cto = system.create_executive_agent("CTO") 
    product_manager = system.create_manager_agent("Product")
    frontend_lead = system.create_team_lead_agent("Frontend Development")
    ui_specialist = system.create_specialist_agent("UI/UX Design")
    
    # Simulate escalation from bottom to top
    print("--- SPECIALIST: Identifies Complex Issue ---")
    specialist_escalation = ui_specialist.invoke({
        "messages": [{"role": "user", "content": "I've discovered a major accessibility compliance issue that affects our entire design system. This requires management decision and significant resources."}]
    })
    specialist_issue = extract_clean_content(specialist_escalation['messages'][-1].content)
    print(f"UI Specialist Escalation: {specialist_issue[:200]}...\n")
    
    print("--- TEAM LEAD: Evaluates and Escalates ---")
    lead_escalation = frontend_lead.invoke({
        "messages": [{"role": "user", "content": "UI specialist identified accessibility compliance issue requiring design system overhaul. This impacts timeline and resources significantly."}]
    })
    lead_assessment = extract_clean_content(lead_escalation['messages'][-1].content)
    print(f"Frontend Lead Assessment: {lead_assessment[:200]}...\n")
    
    print("--- MANAGER: Strategic Assessment ---") 
    manager_escalation = product_manager.invoke({
        "messages": [{"role": "user", "content": "Team lead escalated major accessibility compliance issue requiring design system changes. Needs executive decision on priority and resource allocation."}]
    })
    manager_recommendation = extract_clean_content(manager_escalation['messages'][-1].content)
    print(f"Product Manager Recommendation: {manager_recommendation[:200]}...\n")
    
    print("--- EXECUTIVE: Strategic Decision ---")
    executive_decision = cto.invoke({
        "messages": [{"role": "user", "content": "Product manager escalated accessibility compliance issue requiring significant design system changes. Make strategic decision on priority, timeline, and resource allocation."}]
    })
    executive_resolution = extract_clean_content(executive_decision['messages'][-1].content)
    print(f"CTO Decision: {executive_resolution[:200]}...\n")

def demonstrate_cross_functional_coordination():
    """Demonstrate coordination across multiple departments."""
    print("=== Cross-Functional Coordination ===\n")
    
    system = HierarchicalAgentSystem()
    
    # Create multi-departmental structure
    vp_operations = system.create_executive_agent("VP of Operations")
    engineering_manager = system.create_manager_agent("Engineering")
    marketing_manager = system.create_manager_agent("Marketing")
    sales_manager = system.create_manager_agent("Sales")
    
    # Simulate cross-functional project coordination
    print("--- EXECUTIVE: Cross-Departmental Initiative ---")
    coordination_result = vp_operations.invoke({
        "messages": [{"role": "user", "content": "We're launching a major product update that requires coordination between Engineering, Marketing, and Sales. Coordinate a unified launch strategy."}]
    })
    coordination_plan = extract_clean_content(coordination_result['messages'][-1].content)
    print(f"VP Operations Coordination: {coordination_plan[:200]}...\n")
    
    # Each department responds to coordination
    departments = [
        ("Engineering", engineering_manager, "technical delivery and product readiness"),
        ("Marketing", marketing_manager, "campaign strategy and market positioning"), 
        ("Sales", sales_manager, "sales enablement and customer outreach")
    ]
    
    for dept_name, manager, focus_area in departments:
        print(f"--- {dept_name.upper()} RESPONSE ---")
        dept_result = manager.invoke({
            "messages": [{"role": "user", "content": f"VP Operations is coordinating product launch requiring {focus_area}. Provide departmental plan and coordination requirements."}]
        })
        dept_response = extract_clean_content(dept_result['messages'][-1].content)
        print(f"{dept_name} Manager: {dept_response[:150]}...\n")

if __name__ == "__main__":
    print("🏢 Hierarchical Multi-Agent System")
    print("This system demonstrates organizational agent hierarchies with delegation and escalation\n")
    
    demonstrate_hierarchical_task_flow()
    demonstrate_escalation_flow() 
    demonstrate_cross_functional_coordination()
    
    print("✅ Hierarchical System Benefits:")
    print("   🏢 Organizational structure modeling")
    print("   📈 Scalable delegation patterns")
    print("   🚨 Structured escalation paths")
    print("   🤝 Cross-functional coordination")
    print("   👔 Authority and responsibility clarity")