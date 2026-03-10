# Life-cycle Context
- Control what happens between the core agent steps - intercepting data flow to implement cross-cutting concerns like summarization, guardrails, and logging.
- As you’ve seen in Model Context and Tool Context, middleware is the mechanism that makes context engineering practical. Middleware allows you to hook into any step in the agent lifecycle and either:
    - Update context - Modify state and store to persist changes, update conversation history, or save insights
    - Jump in the lifecycle - Move to different steps in the agent cycle based on context (e.g., skip tool execution if a condition is met, repeat model call with modified context)

![alt text](agent_lifecycle_flow.png)

--- 

## Example: Summarization
One of the most common life-cycle patterns is automatically condensing conversation history when it gets too long. Unlike the transient message trimming shown in Model Context, summarization persistently updates state - permanently replacing old messages with a summary that’s saved for all future turns.
LangChain offers built-in middleware for this:
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        ),
    ],
)
```

- When the conversation exceeds the token limit, SummarizationMiddleware automatically:
    - Summarizes older messages using a separate LLM call
    - Replaces them with a summary message in State (permanently)
    - Keeps recent messages intact for context
The summarized conversation history is permanently updated - future turns will see the summary instead of the original messages.