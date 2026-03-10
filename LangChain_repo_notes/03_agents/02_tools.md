## Tools

- Tools give agents the ability to take actions. Agents go beyond simple model-only tool binding by facilitating:
    - Multiple tool calls in sequence (triggered by a single prompt)
    - Parallel tool calls when appropriate
    - Dynamic tool selection based on previous results
    - Tool retry logic and error handling
    - State persistence across tool calls

- If an empty tool list is provided, the agent will consist of a single LLM node without tool-calling capabilities.

### Tool error handling
- To customize how tool errors are handled, use the `@wrap_tool_call` decorator to create middleware:

```python
...
...
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

...
...

create_agent(
    # ...
    # ...
    middleware=[handle_tool_errors]
)

```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main3_tool_error_handling.py
```


- The agent will return a ToolMessage with the custom error message when a tool fails.

- When no system_prompt is provided, the agent will infer its task from the messages directly.