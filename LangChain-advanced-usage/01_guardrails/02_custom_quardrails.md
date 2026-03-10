# Custom Guardrails

For more sophisticated guardrails, you can create custom middleware that runs before or after the agent executes. This gives you full control over validation logic, content filtering, and safety checks.
​
## Before agent guardrails
Use “before agent” hooks to validate requests once at the start of each invocation. This is useful for session-level checks like authentication, rate limiting, or blocking inappropriate requests before any processing begins.

```python
from typing import Any

from langchain.agents.middleware import before_agent, AgentState, hook_config
# before_agent → middleware that runs BEFORE the agent does *anything*
#                 (before LLM reasoning, before tools, before routing)
#
# AgentState → holds the input messages + context seen by the agent.
# hook_config → supports advanced hook routing (not used here).

from langgraph.runtime import Runtime
# Runtime → provides access to the execution environment.


# -----------------------------------------------------------
# BANNED KEYWORD LIST (OUR GUARDRAIL RULESET)
# -----------------------------------------------------------
banned_keywords = ["hack", "exploit", "malware"]


# -----------------------------------------------------------
# DEFINE THE GUARDRAIL HOOK
# -----------------------------------------------------------
@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Deterministic guardrail:
    Block unsafe or inappropriate user inputs BEFORE the agent begins processing.
    """

    # If for some reason no messages exist, do nothing
    if not state["messages"]:
        return None

    # Look at the FIRST user message only (initial request)
    first_message = state["messages"][0]

    # We only want to filter *human* user inputs
    if first_message.type != "human":
        return None

    # Convert content to lowercase for matching
    content = first_message.content.lower()

    # -------------------------------------------------------
    # CHECK FOR BANNED KEYWORDS
    # -------------------------------------------------------
    for keyword in banned_keywords:
        if keyword in content:
            # If unsafe content is found:
            # 1. Replace output with a safe assistant response
            # 2. Jump execution directly to the "end" node
            #
            # This completely bypasses:
            # - LLM reasoning
            # - Tool usage
            # - Agent logic
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "I cannot process requests containing inappropriate content. "
                        "Please rephrase your request."
                    )
                }],
                "jump_to": "end"   # Stop all further processing
            }

    # If no banned keyword found, continue normally
    return None


# -----------------------------------------------------------
# USE THE CUSTOM GUARDRIAIL IN AN AGENT
# -----------------------------------------------------------
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",                     # Main LLM
    tools=[search_tool, calculator_tool],  # Ordinary tools
    middleware=[content_filter],        # Attach the guardrail
)


# -----------------------------------------------------------
# TEST: THIS INPUT SHOULD BE BLOCKED
# -----------------------------------------------------------
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})

# Agent immediately returns the guardrail error message.
```

---

### 🧱 Deterministic Guardrail

Runs *before* the agent interprets or processes anything.

### 🚫 Blocks unsafe content

Keywords like `"hack"`, `"exploit"`, `"malware"` immediately trigger:

* a safe assistant response
* `jump_to="end"` → full termination of the agent run

No LLM reasoning. No tools. No hallucination.

### ⚡ Fast + predictable

No model call needed — this is pure Python logic.

### 🛡️ Use Cases

Perfect for enterprise/government agents requiring:

* content safety
* policy compliance
* zero-tolerance rules
* predictable pre-screening


### What end means?
**this `"end"` refers to the built-in `END` node of the LangGraph execution graph.**

LangChain v1 agents are internally built **on top of LangGraph**.

Every LangGraph includes two special nodes:

* **START**
* **END**

These are automatically inserted into the graph and require no manual definition.

### So when you write:

```python
@after_agent(can_jump_to=["end"])
```

You are telling LangGraph:

> “This middleware is allowed to jump directly to the **END node** of the underlying graph.”

This is exactly the same `END` in LangGraph diagrams like:

```
START → model → tools → model → END
```

---

- 🧠 Why this matters

Jumping to `"end"` means:

* Immediately terminate agent execution
* Skip all remaining nodes
* Return the currently provided messages as the final output
* Do not call tools or LLM again

This is how guardrails abort the workflow safely.

Internally, LangChain agents compile into a LangGraph `StateGraph`:

* The “agent runner” is just a LangGraph graph
* `"end"` maps directly to the graph’s **terminal state**
* Middleware can redirect execution to this node

Equivalent to:

```python
from langgraph.constants import END
```

But in agent middleware, `"end"` is referenced as a string for convenience.

---

- 📌 Visual

```
.START
   ↓
 model
   ↓
 tools?
   ↓
 model
   ↓
.END   ←──── jump_to="end"
```

---



## After agent guardrails
Use “after agent” hooks to validate final outputs once before returning to the user. This is useful for model-based safety checks, quality validation, or final compliance scans on the complete agent response.


```python
from langchain.agents.middleware import after_agent, AgentState, hook_config
# after_agent → middleware that runs AFTER the agent finishes processing
#                (after LLM reasoning + after all tool calls).
#
# Perfect for checking the *final* output before returning it to the user.

from langgraph.runtime import Runtime
# Runtime → execution environment for LangGraph/LangChain agents.

from langchain_core.messages import AIMessage
# AIMessage → represents model-generated messages.

from langchain.chat_models import init_chat_model
# init_chat_model → lightweight helper for initializing LLMs.

from typing import Any


# -----------------------------------------------------------
# SAFETY EVALUATION MODEL
# -----------------------------------------------------------
# We use a small, cheap model just for safety classification.
# It’s MUCH cheaper than using gpt-4o for the same guardrail.
safety_model = init_chat_model("gpt-4o-mini")


# -----------------------------------------------------------
# MODEL-BASED GUARDRAIL
# Runs AFTER the agent generates a final response.
# This validates the output using a second LLM.
# -----------------------------------------------------------
@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    # If no messages exist, nothing to check
    if not state["messages"]:
        return None

    # Look at the LAST generated message (the agent's final answer)
    last_message = state["messages"][-1]

    # Only evaluate LLM outputs (ignore tool messages, system, etc.)
    if not isinstance(last_message, AIMessage):
        return None

    # -------------------------------------------------------
    # Ask a smaller model to classify the safety of the output
    # -------------------------------------------------------
    safety_prompt = f"""
    Evaluate if this response is safe and appropriate.
    Respond with only 'SAFE' or 'UNSAFE'.

    Response: {last_message.content}
    """

    # The mini-model returns SAFE or UNSAFE deterministically
    result = safety_model.invoke([
        {"role": "user", "content": safety_prompt}
    ])

    # -------------------------------------------------------
    # If unsafe → override final output + stop execution
    # -------------------------------------------------------
    if "UNSAFE" in result.content:
        return {
            "messages": [{
                "role": "assistant",
                "content": (
                    "I cannot provide that response. "
                    "Please rephrase your request."
                )
            }],
            "jump_to": "end",   # Terminate the run immediately
        }

    # Otherwise, return final answer normally
    return None


# -----------------------------------------------------------
# INTEGRATE THE GUARDRAIL WITH AN AGENT
# -----------------------------------------------------------
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",                         # Main reasoning model
    tools=[search_tool, calculator_tool],   # Tools available
    middleware=[safety_guardrail],          # Apply model-based guardrail
)


# -----------------------------------------------------------
# TEST WITH UNSAFE USER REQUEST
# The guardrail will catch and block it.
# -----------------------------------------------------------
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I make explosives?"}]
})
```

---

### 🧠 **Model-Based Guardrails (LLM-as-a-Judge)**

Instead of static keyword filters, we use a **second LLM** to review the output:

* More nuanced
* Better at context
* Can detect subtle harmful content
* Can adapt to policy changes via prompt tuning

### 🔒 **Runs After the Agent Finishes**

`after_agent` ensures:

* Tools may run
* LLM generates its answer
* Final output is intercepted
* Guardrail model reviews safety
* Unsafe answer is replaced

### ✔️ **Uses a Cheap “Safety Model”**

The small model (`gpt-4o-mini`) is:

* inexpensive
* fast
* accurate enough for SAFE/UNSAFE classification

### 🛑 **Unsafe Responses Are Blocked**

Execution jumps to `"end"` and returns a safe fallback message.


## Combine multiple guardrails
You can stack multiple guardrails by adding them to the middleware array. They execute in order, allowing you to build layered protection:

