# Multi-Agent Systems

Multi-agent systems break complex applications into multiple specialized agents that work together to solve problems. Instead of relying on a single agent to handle every step, multi-agent architectures allow you to compose smaller, focused agents into coordinated workflows.

Multi-agent systems are useful when:
- A single agent has too many tools and makes poor decisions about which to use
- Context or memory grows too large for one agent to track effectively  
- Tasks require specialization (e.g., a planner, researcher, math expert)

## 🛠️ Prerequisites

```bash
# Install required packages
pip install langchain python-dotenv

# Set Google API key
export GOOGLE_API_KEY="your-api-key-here"
```

## 📁 Examples

### 1️⃣ [main1_tool_calling_pattern.py](./main1_tool_calling_pattern.py)
**Tool Calling Multi-Agent Pattern**

Shows the centralized control pattern where a supervisor agent calls other agents as tools:
- Supervisor agent coordinates specialized agents (researcher, analyzer, planner)
- Centralized control flow for task orchestration
- Structured task breakdown and execution
- Reusable agent components

```python
# Example: Supervisor delegates to specialized agents
supervisor_agent = create_agent(
    model,
    tools=[research_agent, analysis_agent, planning_agent],
    system_prompt="You coordinate specialized agents to solve complex problems..."
)
```

### 2️⃣ [main2_handoffs_pattern.py](./main2_handoffs_pattern.py)  
**Handoffs Multi-Agent Pattern**

Demonstrates decentralized control where agents transfer control to each other:
- Dynamic expertise routing based on conversation context
- Agents can interact directly with users
- Flexible conversation flow management
- Specialized agent personalities (technical, sales, support)

```python
# Example: Agents handoff control based on expertise
@tool
def handoff_to_technical_specialist(context: str) -> str:
    """Transfer control to technical specialist for technical questions."""
    return f"HANDOFF: Transferring to technical specialist..."
```

### 3️⃣ [main3_collaborative_research.py](./main3_collaborative_research.py)
**Collaborative Multi-Agent Research**

Shows parallel information gathering and synthesis:
- Multiple research perspectives (academic, industry, news)
- Parallel information collection
- Unified synthesis from diverse sources
- Comprehensive analysis workflows

```python
# Example: Parallel research by specialized agents
research_system = CollaborativeResearchSystem()
results = research_system.conduct_collaborative_research("machine learning")
```

### 4️⃣ [main4_agent_debate_consensus.py](./main4_agent_debate_consensus.py)
**Agent Debate and Consensus System**

Demonstrates structured argumentation and consensus building:
- Agents with different perspectives debate issues
- Structured argumentation and counter-argumentation  
- Moderated consensus seeking
- Balanced decision making from multiple viewpoints

```python
# Example: Debate between agents with different perspectives
debate_system = AgentDebateSystem()
result = debate_system.conduct_debate(
    "Should AI companies disclose training data?",
    agents_config=[
        ("Privacy_Advocate", "privacy protection", "privacy law"),
        ("Innovation_Supporter", "technological progress", "AI development")
    ]
)
```

### 5️⃣ [main5_workflow_orchestration.py](./main5_workflow_orchestration.py)
**Workflow Orchestration System**

Shows complex business process automation:
- Sequential and parallel task execution
- Task dependency management
- Error handling and recovery
- Multi-agent coordination for workflows

```python
# Example: Multi-phase workflow execution
orchestrator = WorkflowOrchestrator()
workflow = {
    'id': 'user_onboarding',
    'tasks': [
        {'id': 'create_account', 'dependencies': []},
        {'id': 'send_email', 'dependencies': ['create_account']},
        {'id': 'setup_profile', 'dependencies': ['create_account']}
    ]
}
results = orchestrator.execute_workflow(workflow)
```

### 6️⃣ [main6_hierarchical_agents.py](./main6_hierarchical_agents.py)
**Hierarchical Agent System**

Demonstrates organizational agent structures:
- Executive, manager, team lead, and specialist levels
- Delegation and escalation patterns
- Organizational authority modeling
- Cross-functional coordination

```python
# Example: Hierarchical task delegation
ceo = system.create_executive_agent("CEO")
manager = system.create_manager_agent("Engineering")
team_lead = system.create_team_lead_agent("AI Development")
specialist = system.create_specialist_agent("Machine Learning")
```

## 🎯 Multi-Agent Patterns

| Pattern | Control Flow | Best For | Example Use Case |
|---------|-------------|----------|-----------------|
| **Tool Calling** | Centralized | Task orchestration | Complex analysis workflows |
| **Handoffs** | Decentralized | Domain expertise routing | Customer service systems |
| **Collaborative** | Parallel | Research and analysis | Market research projects |
| **Debate** | Structured | Decision making | Policy evaluation |
| **Workflow** | Sequential/Parallel | Business processes | Product development |
| **Hierarchical** | Organizational | Enterprise systems | Corporate automation |

## 🚦 When to Use Multi-Agent vs LangGraph

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| Quick prototype, simple collaboration | **Multi-Agent** | Easier setup, less complexity |
| Production workflows with persistence | **LangGraph** | Better state management, reliability |
| Complex error handling needed | **LangGraph** | Built-in checkpoints and recovery |
| Human-in-the-loop required | **LangGraph** | Native approval workflows |
| Simple task delegation | **Multi-Agent** | Sufficient for basic coordination |

## 🎨 Design Principles

### 1. Agent Specialization
```python
# Good: Focused agent with clear responsibility
research_agent = create_agent(
    model,
    tools=[academic_search, expert_opinions],
    system_prompt="You are a research specialist focused on academic sources..."
)

# Avoid: General agent with too many responsibilities
```

### 2. Clear Communication Protocols
```python
# Good: Structured handoff with context
@tool
def handoff_to_specialist(context: str, urgency: str) -> str:
    """Transfer control with clear context and priority."""
    return f"HANDOFF: {context} | Priority: {urgency}"
```

### 3. Error Handling and Recovery
```python
# Good: Graceful error handling with recovery
@tool
def handle_task_failure(task_id: str, error: str, recovery: str) -> str:
    """Handle failures with appropriate recovery actions."""
    return f"ERROR HANDLING: {task_id} | Recovery: {recovery}"
```

## 🔧 Implementation Tips

1. **Start Simple**: Begin with tool calling pattern before moving to complex coordination
2. **Define Clear Roles**: Each agent should have specific, well-defined responsibilities  
3. **Handle State Carefully**: Multi-agent systems can lose context without proper state management
4. **Plan for Errors**: Include error handling and recovery in your agent interactions
5. **Monitor Performance**: Track agent interactions and identify bottlenecks
6. **Test Incrementally**: Add agents one at a time and test interactions

## ⚖️ Trade-offs

**Advantages:**
- ✅ Specialized expertise and focused capabilities
- ✅ Scalable through agent composition
- ✅ Flexible coordination patterns
- ✅ Parallel processing capabilities
- ✅ Clear separation of concerns

**Challenges:**
- ❌ Coordination complexity increases with scale
- ❌ State management becomes difficult
- ❌ Error handling across agents is complex
- ❌ Debugging distributed agent interactions
- ❌ Performance overhead from agent communication

## 🎯 Choose Your Pattern

1. **Tool Calling** → Start here for most use cases
2. **Handoffs** → When you need dynamic expertise routing
3. **Collaborative** → For parallel information processing
4. **Debate** → When multiple perspectives add value
5. **Workflow** → For business process automation
6. **Hierarchical** → For enterprise organizational modeling