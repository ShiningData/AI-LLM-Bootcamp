# Human-in-the-Loop

The Human-in-the-Loop (HITL) middleware lets you add human oversight to agent tool calls. When a model proposes an action that might require review — for example, writing to a file or executing SQL — the middleware can pause execution and wait for a decision.

It does this by checking each tool call against a configurable policy. If intervention is needed, the middleware issues an interrupt that halts execution. The graph state is saved using LangGraph's persistence layer, so execution can pause safely and resume later.

A human decision then determines what happens next: the action can be approved as-is (approve), modified before running (edit), or rejected with feedback (reject).

These examples show how to:
- Configure Human-in-the-Loop middleware with per-tool policies
- Handle approve/edit/reject decisions in different scenarios
- Implement conditional approval based on argument values
- Manage multi-step approval workflows
- Set up persistent checkpointing for production use

## 🛠️ Prerequisites

```bash
# Install required packages
pip install langchain python-dotenv

# Set Google API key (for examples that use LangChain agents)
export GOOGLE_API_KEY="your-api-key-here"
```

## 📁 Examples

### 1️⃣ [main1_basic_hitl_approval.py](./main1_basic_hitl_approval.py)
**Basic Human-in-the-Loop Approval**

**What it is**: Demonstrates basic HITL workflow with different approval policies for safe vs dangerous tools.

**How to run**:
```bash
uv --project uv_env/ run python main1_basic_hitl_approval.py
```

**What to expect**:
```
=== Basic Human-in-the-Loop Approval Example ===

🔍 Step 1: Testing safe operation (read_user_data)
This should execute immediately without interruption...

✅ Safe operation completed:
Result: User data for 12345: Name=John Doe, Email=john@example.com, Status=Active

🚨 Step 2: Testing dangerous operation (delete_user_account)  
This should be interrupted and require human approval...

🛑 INTERRUPT DETECTED!
Action requests:
  - Tool: delete_user_account
  - Arguments: {'user_id': '12345'}
  - Description: Tool execution requires human approval

Human decision: REJECT (too dangerous!)

✅ Execution resumed with REJECT decision:
Result: Account deletion was rejected for safety reasons
```

**Key concepts**: Basic approve/reject workflow, safe vs dangerous tool policies

---

### 2️⃣ [main2_edit_tool_arguments.py](./main2_edit_tool_arguments.py)
**Tool Argument Editing**

**What it is**: Shows how humans can edit tool arguments before execution to make operations safer.

**How to run**:
```bash
uv --project uv_env/ run python main2_edit_tool_arguments.py
```

**What to expect**:
```
=== Human-in-the-Loop Tool Argument Editing Example ===

📝 Step 1: File writing with argument editing
Requesting to write a potentially problematic file...

🛑 INTERRUPT: File writing operation requires approval
Original request:
  - Tool: write_file
  - Arguments: {'filepath': '/etc/passwd', 'content': 'Hello World'}

🔧 Human decision: EDIT (change to safe location)

✅ File operation executed with edited arguments:
Result: File written to /tmp/hello.txt with 29 characters
```

**Key concepts**: Argument editing, different policies per tool, security through modification

---

### 3️⃣ [main3_conditional_approval.py](./main3_conditional_approval.py)
**Conditional Approval Logic**

**What it is**: Implements conditional approval based on argument values and business rules.

**How to run**:
```bash
uv --project uv_env/ run python main3_conditional_approval.py
```

**What to expect**:
```
=== Conditional Human-in-the-Loop Approval Example ===

💰 Test 1: Small transfer (should be low risk)
Transfer $50 from internal account...

🛑 Transfer interrupted for review:
  Amount: $50.0
  To account: account_456
✅ AUTO-APPROVED: Small internal transfer

💰 Test 2: Large transfer (should require approval)

🛑 Large transfer interrupted:
  Amount: $5000.0
  To account: external_bank_999
🚨 HIGH RISK: Large amount + external account
❌ Human decision: REJECT
```

**Key concepts**: Dynamic approval rules, risk-based decisions, business logic integration

---

### 4️⃣ [main4_multi_step_approval.py](./main4_multi_step_approval.py)
**Multi-Step Approval Workflows**

**What it is**: Handles complex workflows with multiple approval points and cascading decisions.

**How to run**:
```bash
uv --project uv_env/ run python main4_multi_step_approval.py
```

**What to expect**:
```
=== Multi-Step Human-in-the-Loop Approval Example ===

🚀 Deployment Request: Deploy new application version with database changes

🛑 DEPLOYMENT STEP 1 - Approval Required
==================================================

📋 Action 1: create_database_backup
Arguments: {'database_name': 'main_db'}
💭 Human Review: Database backup is safe and necessary
✅ Decision: APPROVE

📋 Action 2: apply_schema_migration
Arguments: {'database_name': 'main_db', 'migration_file': 'migration_v2.1.sql'}
💭 Human Review: Migration file needs verification
✅ Decision: APPROVE
```

**Key concepts**: Multi-step workflows, cascading approvals, deployment scenarios

---

### 5️⃣ [main5_persistent_checkpointing.py](./main5_persistent_checkpointing.py)
**Persistent Checkpointing**

**What it is**: Demonstrates production-ready persistent checkpointing for long-running approval processes.

**How to run**:
```bash
uv --project uv_env/ run python main5_persistent_checkpointing.py
```

**What to expect**:
```
=== Persistent Checkpointing HITL Example ===

🏗️ Step 1: Creating approval requests from multiple users
============================================================

👤 User alice (Thread: finance_001)
Task: Process payment of $5000 to vendor ABC Corp
🛑 Approval required for process_payment
   Request ID: finance_001_1733108887
   Status: Pending approval

⚖️ Step 3: Process approvals (simulating human reviewers)
============================================================

👨‍💼 supervisor_jane reviewing request finance_001_1733108887
   Tool: process_payment
   Decision: APPROVE
   Feedback: Payment approved for legitimate vendor
```

**Key concepts**: Persistent state, multi-user workflows, production checkpointing

## 🚀 Quick Start Guide

**Want to see Human-in-the-Loop in action? Follow these steps:**

1. **Set your Google API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **Start with the basic example**:
   ```bash
   uv --project uv_env/ run python main1_basic_hitl_approval.py
   ```

3. **Watch the workflow**: The agent will demonstrate:
   - Safe operations executing immediately
   - Dangerous operations being interrupted for approval
   - Human decisions (approve/reject) affecting execution
   - State preservation during interrupts

**That's it!** You've seen the basic HITL workflow in action.

**Next steps:**
- Try argument editing: `python main2_edit_tool_arguments.py`
- Explore conditional logic: `python main3_conditional_approval.py`
- See complex workflows: `python main4_multi_step_approval.py`
- Test production features: `python main5_persistent_checkpointing.py`

## 📖 Key Learning Points

### 1. **Approval Policies**
- Configure per-tool policies: `True`, `False`, or `{"allowed_decisions": [...]}`
- Safe tools can auto-execute, dangerous tools require approval
- Different tools can have different policies

### 2. **Human Decisions**
- **Approve**: Execute the tool as originally requested
- **Edit**: Modify arguments before execution
- **Reject**: Stop execution with feedback

### 3. **Checkpointing Requirements**
- Must use a checkpointer for interrupt/resume workflows
- Thread ID required for persistent state
- Production needs persistent checkpointers (PostgreSQL, Redis, etc.)

### 4. **Workflow Patterns**
- Simple approve/reject for basic safety
- Argument editing for security modifications  
- Conditional approval based on business rules
- Multi-step workflows for complex processes

## 🔧 Troubleshooting

### Common Issues

1. **Missing Thread ID**
   ```bash
   # Always provide thread_id in config
   config = {"configurable": {"thread_id": "unique_id"}}
   ```

2. **No Checkpointer**
   ```python
   # HITL requires checkpointing
   agent = create_agent(model, tools, middleware=[...], checkpointer=InMemorySaver())
   ```

3. **Google API Key Missing**
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

4. **Mismatched Decisions**
   - Number of decisions must match number of interrupted actions
   - Decision types must match allowed_decisions policy

### Debug Tips

- Check interrupt payload for available decision types
- Verify thread_id consistency between invoke and resume
- Use `verbose=True` in agent creation for detailed logs
- Test with InMemorySaver before production checkpointers

## 📚 Resources

- [LangChain Human-in-the-Loop Documentation](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangGraph Interrupts Guide](https://langchain-ai.github.io/langgraph/concepts/interrupts/)
- [Checkpointing Documentation](https://langchain-ai.github.io/langgraph/concepts/checkpointing/)

## 🎯 Next Steps

After working through these examples:

1. **Configure your own approval policies** based on your security requirements
2. **Implement production checkpointing** with persistent storage
3. **Build approval dashboards** for human reviewers
4. **Add audit logging** for compliance and governance
5. **Integrate with your authorization system** for role-based approvals

## Configuring interrupts

- **fine-grained human approval rules per tool** using `HumanInTheLoopMiddleware`.
- Human-in-the-Loop With Per-Tool Rules

```python
from langchain.agents import create_agent
# create_agent → builds LangChain v1 agent (backed by LangGraph)

from langchain.agents.middleware import HumanInTheLoopMiddleware
# HumanInTheLoopMiddleware → pauses agent execution when a tool requires approval.
# Allows humans to: approve / reject / edit tool calls.

from langgraph.checkpoint.memory import InMemorySaver
# InMemorySaver → stores agent state so it can pause + resume during interrupts.
# Required for Human-in-the-Loop workflows.


# -----------------------------------------------------------
# CREATE AGENT WITH HUMAN-IN-THE-LOOP APPROVAL RULES
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",    # Main LLM

    tools=[
        write_file_tool,     # potentially destructive → approval required
        execute_sql_tool,    # dangerous → approval but no edits allowed
        read_data_tool,      # safe → no approval needed
    ],

    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # ---------------------------------------------------
                # 1) write_file: Full human control
                # ---------------------------------------------------
                "write_file": True,
                # Meaning:
                #   - Interrupt ALWAYS before calling write_file
                #   - Human may approve, reject, or edit the tool input

                # ---------------------------------------------------
                # 2) execute_sql: Restricted human control
                # ---------------------------------------------------
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"]
                },
                # Meaning:
                #   - Interrupt before execute_sql
                #   - Human can approve or reject
                #   - Editing SQL queries is *not allowed* (safer)

                # ---------------------------------------------------
                # 3) read_data: No interrupt (always auto-approved)
                # ---------------------------------------------------
                "read_data": False
                # Meaning:
                #   - Agent executes read-only tool automatically
                #   - No pause, no human action required
            },

            # -------------------------------------------------------
            # OPTIONAL: CUSTOM PREFIX FOR INTERRUPT MESSAGES
            # -------------------------------------------------------
            # This text appears when agent pauses for approval:
            # Example:
            #   "Tool execution pending approval: execute_sql with query='DELETE FROM ...'"
            description_prefix="Tool execution pending approval",
        ),
    ],

    # -----------------------------------------------------------
    # CHECKPOINTING REQUIRED FOR HUMAN-IN-THE-LOOP WORKFLOWS
    # -----------------------------------------------------------
    checkpointer=InMemorySaver(),
    # In production replace with:
    #   AsyncPostgresSaver
    #   RedisSaver
    #   DynamoDBSaver
    # So human approval survives restarts and multiple workers.
)
```

---

### ✔️ 1. **Per-tool approval rules**

Each tool has its own policy:

| Tool          | Approval? | Edit allowed? |
| ------------- | --------- | ------------- |
| `write_file`  | Yes       | Yes           |
| `execute_sql` | Yes       | ❌ No edits    |
| `read_data`   | No        | N/A           |

This pattern is critical for:

* security
* compliance
* database-safety
* production AI systems

---

### ✔️ 2. **Interrupt + Resume workflow**

The agent:

1. Detects tool call
2. Pauses execution
3. Emits interrupt message
4. Stores state (checkpoint)
5. Waits for human approval
6. Resumes from checkpoint

This is the standard **Human-in-the-Loop LangGraph pattern**.

---

### ✔️ 3. **Safety by design**

Dangerous tools like:

* database writes
* SQL execution
* file writes
* external API calls
* cloud provisioning

should *always* go through Human-in-the-Loop.

---

### ✔️ 4. **Checkpointing is mandatory**

Without checkpointing, the agent cannot pause/resume mid-run.

- You must configure a checkpointer to persist the graph state across interrupts. In production, use a persistent checkpointer like AsyncPostgresSaver. For testing or prototyping, use InMemorySaver.
- When invoking the agent, pass a config that includes the thread ID to associate execution with a conversation thread. See the LangGraph interrupts documentation for details.

## Responding to interrupts
- When you invoke the agent, it runs until it either completes or an interrupt is raised. An interrupt is triggered when a tool call matches the policy you configured in interrupt_on. In that case, the invocation result will include an __interrupt__ field with the actions that require review. You can then present those actions to a reviewer and resume execution once decisions are provided.

- **interrupts**, **thread_id**, **Command(resume=...)**, and how LangGraph pauses/resumes execution.
- HITL Interrupt + Resume Flow Using Command

```python
from langgraph.types import Command
# Command → Special LangGraph instruction used to resume execution
# after an interrupt triggered by HumanInTheLoopMiddleware.


# -----------------------------------------------------------
# CONFIGURATION: THREAD ID REQUIRED FOR HITL
# -----------------------------------------------------------
config = {"configurable": {"thread_id": "some_id"}}
"""
Human-in-the-Loop requires persistence between steps.
Why?

1. Agent runs → hits interrupt (needs approval)
2. Execution stops
3. State is saved to checkpointer (InMemorySaver or PostgresSaver)
4. Later: user approves / edits / rejects
5. Agent resumes from EXACT state before interrupt

A stable 'thread_id' ties both steps together.
"""


# -----------------------------------------------------------
# STEP 1 — INVOKE AGENT UNTIL THE INTERRUPT OCCURS
# -----------------------------------------------------------
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config    # must supply thread_id
)
"""
Because the agent attempts to call a risky tool (execute_sql),
HumanInTheLoopMiddleware pauses the run and emits an interrupt.

Execution stops BEFORE the tool actually runs.
"""


# -----------------------------------------------------------
# EXAMINE THE INTERRUPT PAYLOAD
# -----------------------------------------------------------
print(result["__interrupt__"])
"""
The interrupt contains:

- action_requests:  
    Every tool call that requires human review, including:
        - tool name
        - tool arguments
        - description (from middleware)

- review_configs:
    Allowed decisions for each tool:
        - approve
        - reject
        - edit (if allowed)

Example output:
[
    Interrupt(
        value={
            'action_requests': [
                {
                    'name': 'execute_sql',
                    'arguments': {'query': 'DELETE ...'},
                    'description': 'Tool execution pending approval...'
                }
            ],
            'review_configs': [
                {
                    'action_name': 'execute_sql',
                    'allowed_decisions': ['approve', 'reject']
                }
            ]
        }
    )
]
"""
# NOTE:
# The tool has NOT executed yet — this is only a request for approval.


# -----------------------------------------------------------
# STEP 2 — RESUME EXECUTION WITH A HUMAN DECISION
# -----------------------------------------------------------
agent.invoke(
    Command(
        resume={           # Tell LangGraph we’re resuming from interrupt
            "decisions": [
                {
                    "type": "approve"  # Could also be "edit" or "reject"
                }
            ]
        }
    ),
    config=config         # MUST match the same thread_id
)
"""
Command(resume=...) tells LangGraph:

- Continue execution from the paused state
- Apply the provided human decision
- If approved → run the tool
- If rejected → skip / return safe message
- If edited → use edited arguments

Because the same thread_id is used, LangGraph retrieves
the correct saved checkpoint and resumes seamlessly.
"""
```

---

### ✔️ 1. **Interrupts stop execution before dangerous tools run**

The agent NEVER executes:

```sql
DELETE FROM records...
```

unless the human explicitly approves.

---

### ✔️ 2. **Interrupt payload provides full metadata**

Includes:

* the tool call
* arguments
* description
* allowed decisions (approve/reject/edit)

You can display this to the user in any frontend.

---

### ✔️ 3. **`thread_id` links pause → resume**

Without `thread_id`, the agent cannot resume.

---

### ✔️ 4. **`Command(resume=...)` continues execution**

This is how LangGraph moves forward after human approval.

---

### ✔️ 5. **Checkpointing is required**

Your `checkpointer=InMemorySaver()` handles persistence for the demo.
Production requires:

* `AsyncPostgresSaver`
* `RedisSaver`
* `DynamoDBSaver`


## Execution lifecycle
- The middleware defines an after_model hook that runs after the model generates a response but before any tool calls are executed:
    - The agent invokes the model to generate a response.
    - The middleware inspects the response for tool calls.
    - If any calls require human input, the middleware builds a HITLRequest with action_requests and review_configs and calls interrupt.
    - The agent waits for human decisions.
    - Based on the HITLResponse decisions, the middleware executes approved or edited calls, synthesizes ToolMessage’s for rejected calls, and resumes execution.

## Custom HITL logic
For more specialized workflows, you can build custom HITL logic directly using the interrupt primitive and middleware abstraction.
Review the execution lifecycle above to understand how to integrate interrupts into the agent’s operation.