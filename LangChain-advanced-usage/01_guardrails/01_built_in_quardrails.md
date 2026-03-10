## Guardrails
- Implement safety checks and content filtering for your agents

- Guardrails help you build safe, compliant AI applications by validating and filtering content at key points in your agent’s execution. They can detect sensitive information, enforce content policies, validate outputs, and prevent unsafe behaviors before they cause problems.
Common use cases include:
    - Preventing PII leakage
    - Detecting and blocking prompt injection attacks
    - Blocking inappropriate or harmful content
    - Enforcing business rules and compliance requirements
    - Validating output quality and accuracy
You can implement guardrails using middleware to intercept execution at strategic points - before the agent starts, after it completes, or around model and tool calls.

![alt text](image.png)

---

Guardrails can be implemented using two complementary approaches:

### 1. Deterministic guardrails
Use rule-based logic like regex patterns, keyword matching, or explicit checks. Fast, predictable, and cost-effective, but may miss nuanced violations.

### 2. Model-based guardrails
Use LLMs or classifiers to evaluate content with semantic understanding. Catch subtle issues that rules miss, but are slower and more expensive.

## Built-in guardrails
​
### PII detection
LangChain provides built-in middleware for detecting and handling Personally Identifiable Information (PII) in conversations. This middleware can detect common PII types like emails, credit cards, IP addresses, and more.
PII detection middleware is helpful for cases such as health care and financial applications with compliance requirements, customer service agents that need to sanitize logs, and generally any application handling sensitive user data.
The PII middleware supports multiple strategies for handling detected PII:
| Strategy   | Behavior                                     | Example Input         | Example Output        |
| ---------- | -------------------------------------------- | --------------------- | --------------------- |
| **redact** | Replace with placeholder `[REDACTED_TYPE]`   | `john@example.com`    | `[REDACTED_EMAIL]`    |
| **mask**   | Partially obscure data (keep last 4 digits)  | `4532-1234-5678-9010` | `****-****-****-9010` |
| **hash**   | Replace with deterministic irreversible hash | `+1-555-123-4567`     | `a8f5f1676f...`       |
| **block**  | Raise exception if detected                  | `sk-123abc…`          | ❌ error thrown        |


```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


agent = create_agent(
    model="gpt-4o",
    tools=[customer_service_tool, email_tool],
    middleware=[

        # ---------------------------------------------------
        # EMAILS → REDACT
        # ---------------------------------------------------
        # Replaces detected emails with standardized placeholder:
        #   john@example.com → [REDACTED_EMAIL]
        PIIMiddleware(
            "email",
            strategy="redact",       # Replace with [REDACTED_EMAIL]
            apply_to_input=True,
        ),

        # ---------------------------------------------------
        # CREDIT CARDS → MASK
        # ---------------------------------------------------
        # Partially hides values but keeps format:
        #   4532-1234-5678-9010 → ****-****-****-9010
        PIIMiddleware(
            "credit_card",
            strategy="mask",         # Keep last 4 digits visible
            apply_to_input=True,
        ),

        # ---------------------------------------------------
        # API KEYS → BLOCK
        # ---------------------------------------------------
        # If detected, the middleware raises an exception and stops execution.
        #   sk-abc123... → Error thrown
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",        # Do not allow request to proceed
            apply_to_input=True,
        ),

        # ---------------------------------------------------
        # PHONE NUMBERS → HASH (DETERMINISTIC)
        # ---------------------------------------------------
        # Converts the detected PII into a deterministic hash:
        #   +1-555-123-4567 → a8f5f1676f...
        #
        # This preserves uniqueness for analytics while protecting identity.
        PIIMiddleware(
            "phone",
            strategy="hash",         # Deterministic irreversible hash
            apply_to_input=True,
        ),
    ],
)

# Invocation
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "My email is john@example.com, phone is +1-555-123-4567, "
            "and my card number is 4532-1234-5678-9010"
        )
    }]
})
```

## Human-in-the-loop
- Built-in middleware for requiring human approval before executing sensitive operations. This is one of the most effective guardrails for high-stakes decisions.
- Helpful for cases such as financial transactions and transfers, deleting or modifying production data, sending communications to external parties, and any operation with significant business impact.

```python
from langchain.agents import create_agent
# create_agent → builds a LangChain v1 agent with tools + middleware.

from langchain.agents.middleware import HumanInTheLoopMiddleware
# HumanInTheLoopMiddleware →
#   Pauses the agent execution before risky tool calls
#   and waits for explicit human approval (or rejection).

from langgraph.checkpoint.memory import InMemorySaver
# InMemorySaver → persists agent state so paused runs can be resumed later.

from langgraph.types import Command
# Command(resume=...) → used to resume the paused agent workflow.


# -----------------------------------------------------------
# CREATE AGENT WITH HUMAN-IN-THE-LOOP SAFETY
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",   # Main LLM guiding reasoning + tool selection

    tools=[search_tool, send_email_tool, delete_database_tool],
    # Tools available to the agent; some are sensitive

    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval BEFORE executing these tools:
                "send_email": True,         # Sensitive (external effect)
                "delete_database": True,    # Extremely sensitive/destructive

                # Safe operations — no human approval needed:
                "search": False,
            }
        ),
    ],

    # Persist agent state between "pause" and "resume" calls
    # HITL relies on this persistence to continue from where it left off
    checkpointer=InMemorySaver(),
)


# -----------------------------------------------------------
# A CONFIG WITH THREAD ID IS REQUIRED
# HITL → Must always resume in the same thread
# -----------------------------------------------------------
config = {"configurable": {"thread_id": "some_id"}}


# -----------------------------------------------------------
# 1) FIRST INVOCATION
# Agent decides it needs to call: send_email_tool
#
# HumanInTheLoopMiddleware sees:
#   interrupt_on["send_email"] = True
#
# → Agent pauses execution and returns an "approval required" response.
# -----------------------------------------------------------
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)

# At this point, the agent is PAUSED.
# The result contains metadata describing:
#   - what action is waiting for approval
#   - which tool call is pending
#   - instructions for resuming


# -----------------------------------------------------------
# 2) RESUME EXECUTION WITH HUMAN APPROVAL
# A human (or UI system) sends: approve / reject
#
# Command(resume={...}) continues the previously paused workflow.
# -----------------------------------------------------------
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config   # Must use SAME thread_id to resume state
)
```