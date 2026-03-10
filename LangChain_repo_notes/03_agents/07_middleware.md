
## Middleware
- Middleware provides powerful extensibility for **customizing agent behavior at different stages of execution.** You can use middleware to:
    - Process state before the model is called (e.g., message trimming, context injection)
    - Modify or validate the model’s response (e.g., guardrails, content filtering)
    - Handle tool execution errors with custom logic
    - Implement dynamic model selection based on state or context
    - Add custom logging, monitoring, or analytics
    - Middleware integrates seamlessly into the agent’s execution graph, allowing you to intercept and modify data flow at key points without changing the core agent logic.