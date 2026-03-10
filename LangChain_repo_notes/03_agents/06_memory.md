## Memory
- This refers to how agents maintain conversation history automatically through the message state during a single agent execution langchain. It's essentially short-term memory that includes:
    - Message history - automatically tracked in the agent's state
    - Custom state fields - you can extend AgentState to remember additional info during the conversation

```
python# The agent automatically remembers messages within this run
agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather?"}]
})

# The agent has access to all previous messages in this conversation
```
### This memory only exists during that specific agent run - once the agent finishes, the memory is gone (unless you persist it yourself).

- Information stored in the state can be thought of as the short-term memory of the agent:
- Custom state schemas must extend AgentState as a TypedDict.
- There are two ways to define custom state:
    - 1. Via middleware (preferred)
    - 2. Via state_schema on create_agent

### 1. Defining state via middleware
- Use middleware to define custom state when your custom state needs to be accessed by specific middleware hooks and tools attached to said middleware.
```python
from typing import Any
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langchain.agents import AgentState
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", max_tokens=500)

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Found info about: {query}"

class CustomState(AgentState):
    user_preferences: dict = {}

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [search]

    def before_model(self, state, runtime=None) -> dict[str, Any] | None:
        """
        This method is called automatically BEFORE the AI model processes each message.
        It allows us to modify the conversation or inject additional context.
        """
        # Get user preferences from state - handle both dict and object formats
        # state can be a dict {'user_preferences': {...}} or an object with .user_preferences
        user_prefs = state.get('user_preferences', {}) if isinstance(state, dict) else getattr(state, 'user_preferences', {})
        
        if user_prefs:
            # Convert preferences dict to readable string: {"music": "blues"} -> "music: blues"
            prefs = ", ".join([f"{k}: {v}" for k, v in user_prefs.items()])
            print(f"🔧 Smart assistant reminding about preferences: {prefs}")
            
            if 'messages' in state:
                # Inject system message at the beginning of conversation to remind AI about preferences
                # This happens BEFORE the model sees the messages - that's why it's called "before_model"
                state['messages'].insert(0, {
                    "role": "system", 
                    "content": f"Remember the user's preferences: {prefs}. Use this information when responding."
                })
        return None

def main():
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[CustomMiddleware()]  # Smart assistant that reminds about preferences
    )
    
    print("🔧 Smart Assistant with Preferences")
    print("Type 'quit' to exit or 'prefs' to set preferences\n")
    
    state = {
        "messages": [],  # The notebook (conversation history)
        "user_preferences": {}  # The preferences the smart assistant remembers
    }
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'prefs':
            key = input("Preference key: ")
            value = input("Preference value: ")
            state["user_preferences"][key] = value
            print(f"Smart assistant learned: {key} = {value}\n")
            continue
        
        state["messages"].append({"role": "user", "content": user_input})
        result = agent.invoke(state)
        state = result
        
        agent_message = result["messages"][-1].content
        # Extract just the text if it's in a list format
        if isinstance(agent_message, list) and len(agent_message) > 0:
            agent_message = agent_message[0].get('text', str(agent_message))
        print(f"Agent: {agent_message}\n")

if __name__ == "__main__":
    main()
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main6_memory.py
```

- Example dialog
```
 uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main6_memory.py 
🔧 Smart Assistant with Preferences
Type 'quit' to exit or 'prefs' to set preferences

You: prefs
Preference key: favorite fruit
Preference value: banana
Smart assistant learned: favorite fruit = banana

You: what fruit I like
🔧 Smart assistant reminding about preferences: favorite fruit: banana
Agent: You like bananas.

You: I am Erkan
🔧 Smart assistant reminding about preferences: favorite fruit: banana
Agent: Hello Erkan, nice to meet you! How can I help you today?

You: what is my name?
🔧 Smart assistant reminding about preferences: favorite fruit: banana
Agent: Your name is Erkan.

You: quit
```


- As of langchain 1.0, **custom state schemas must be TypedDict** types. Pydantic models and dataclasses are no longer supported. See the v1 migration guide for more details.



### 2. Defining  state_schema on create_agent
- Use the state_schema parameter as a shortcut to define custom state that is only used in tools.

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```