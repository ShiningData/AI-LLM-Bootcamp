# Best practices
- Start simple - Begin with static prompts and tools, add dynamics only when needed
- Test incrementally - Add one context engineering feature at a time
- Monitor performance - Track model calls, token usage, and latency
- Use built-in middleware - Leverage SummarizationMiddleware, LLMToolSelectorMiddleware, etc.
- Document your context strategy - Make it clear what context is being passed and why
- Understand transient vs persistent: Model context changes are transient (per-call), while life-cycle context changes persist to state