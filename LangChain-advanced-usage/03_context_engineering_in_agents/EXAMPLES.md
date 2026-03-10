# Context Engineering Examples

This directory contains focused examples of LangChain model context engineering - controlling what goes into each model call for optimal reliability and cost.

## Files

### `main1_dynamic_prompts.py`
**System Prompt Engineering**
- **State-aware prompts** based on conversation length
- **Store-based prompts** using user preferences from memory
- **Context-aware prompts** adapting to user roles and environments
- Dynamic system prompt generation with `@dynamic_prompt`

### `main2_message_injection.py`
**Message Context Injection**
- **File context injection** from agent state (uploaded documents)
- **Writing style injection** from user preferences in store
- **Compliance rules injection** based on jurisdiction and industry
- Message modification with `@wrap_model_call` middleware

### `main3_dynamic_tools.py`
**Dynamic Tool Selection**
- **State-based filtering** (authentication, conversation progress)
- **Role-based access control** (RBAC) with department restrictions
- **Permission-based filtering** using stored user permissions
- Runtime tool availability modification

### `main4_model_selection.py`
**Dynamic Model Selection**
- **Conversation length-based** model switching (efficient → standard → large)
- **Task complexity-based** model selection for optimal performance
- **Tool-aware selection** matching model capabilities to available tools
- **User preference-based** selection (cost vs quality trade-offs)

### `main5_response_format.py`
**Dynamic Response Formatting**
- **Pydantic schema definitions** for structured outputs
- **Conversation state-based** format adaptation
- **Role-specific schemas** (support tickets, task plans, analysis reports)
- **Content-aware formatting** based on query type

### `main6_tool_state_reads.py`
**Tool Context Reading**
- **Reading from agent state** (authentication, session info, conversation context)
- **Accessing persistent store** (user preferences, security settings, history)
- **Using runtime context** (environment config, security validation)
- **Comprehensive context checks** combining all sources

### `main7_tool_state_writes.py`
**Tool Context Writing**
- **Writing to state with Commands** (authentication, permissions, session data)
- **Updating persistent store** (user preferences, profiles, activity logs)
- **Combined updates** (state + store in single tool)
- **State persistence** across conversation turns

### `main8_conversation_summarization.py`
**Conversation Summarization**
- **Built-in SummarizationMiddleware** for automatic conversation management
- **Custom summarization logic** with selective message preservation
- **Conversation length monitoring** with before/after hooks
- **Memory persistence** across summarization events

### `main9_lifecycle_hooks.py`
**Lifecycle Hooks**
- **before_agent and after_agent** hooks for request lifecycle
- **before_model and after_model** hooks for model call monitoring
- **Security validation and monitoring** through lifecycle interception
- **Performance tracking** and error handling patterns

### `main10_persistent_vs_transient.py`
**Persistent vs Transient Updates**
- **Transient updates** with @wrap_model_call (temporary modifications)
- **Persistent updates** with before/after hooks (permanent state changes)
- **Mixed strategies** combining both approaches
- **Cross-session persistence** and state management

## Running Examples

Each file can be run independently:

```bash
python main1_dynamic_prompts.py
python main2_message_injection.py  
python main3_dynamic_tools.py
python main4_model_selection.py
python main5_response_format.py
python main6_tool_state_reads.py
python main7_tool_state_writes.py
python main8_conversation_summarization.py
python main9_lifecycle_hooks.py
python main10_persistent_vs_transient.py
```

## Key Concepts

### System Prompts
- `@dynamic_prompt` decorator for runtime prompt generation
- Context-aware instructions based on user roles, preferences, conversation state
- Store integration for personalized prompting

### Messages
- `@wrap_model_call` middleware for message modification
- Context injection (files, writing style, compliance rules)
- Runtime message manipulation before model calls

### Tools
- Dynamic tool filtering based on authentication, roles, permissions
- State-aware tool availability (conversation progress, user capabilities)
- Security and access control through tool restriction

### Model Selection
- Multi-model architecture for cost and performance optimization
- Context-driven model switching (conversation length, task complexity)
- User preference and tool-aware model selection

### Response Format
- Structured output with Pydantic schemas
- Dynamic format selection based on context and user needs
- Role-specific and content-aware output formatting

### Tool Context
- Tools reading from state, store, and runtime context
- Tools writing to state using Command objects
- Authentication and session management through tools
- Persistent data storage and retrieval in tools

### Lifecycle Context
- Conversation summarization for memory management
- Lifecycle hooks for monitoring and instrumentation
- Persistent vs transient context modifications
- Cross-cutting concerns like logging, security, and performance

## Architecture Patterns

- **Middleware-driven**: All context engineering happens through middleware
- **State-aware**: Decisions based on conversation history and agent state
- **Store-integrated**: User preferences and memory inform context decisions
- **Runtime-contextual**: Per-request context shapes model behavior
- **Security-conscious**: Role-based access and compliance integration
- **Lifecycle-aware**: Hooks into agent execution flow for monitoring and modification
- **Persistence-conscious**: Clear distinction between temporary and permanent updates