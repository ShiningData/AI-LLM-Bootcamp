# Runtime Examples

This directory contains focused examples of LangChain runtime features:

## Files

### `main1_basic_context.py`
- **Context schema definition** with `@dataclass`
- **Context access in tools** using `ToolRuntime[Context]`
- **Role-based customization** in tool behavior
- Shows how to pass context when invoking agents

### `main2_dynamic_prompts.py` 
- **Dynamic system prompts** using `@dynamic_prompt`
- **Context-aware prompt generation** based on user preferences
- **Personality and expertise adaptation**
- Access to `ModelRequest.runtime` in middleware

### `main3_lifecycle_hooks.py`
- **Before/after model hooks** using `@before_model` and `@after_model`
- **Request auditing and logging**
- **Security validation** in middleware
- **Performance monitoring** and metrics collection

### `main4_store_memory.py`
- **Long-term memory** with runtime store
- **User-scoped and workspace-scoped** data storage
- **Persistent memory** across sessions
- Simulation of real store operations

## Running Examples

Each file can be run independently:

```bash
python main1_basic_context.py
python main2_dynamic_prompts.py  
python main3_lifecycle_hooks.py
python main4_store_memory.py
```

## Key Concepts

- **Dependency Injection**: Runtime context eliminates global state
- **Personalization**: Tools and prompts adapt to user context
- **Observability**: Lifecycle hooks enable monitoring and auditing
- **Memory**: Store enables persistent data across sessions
- **Testability**: Context injection makes tools easier to test