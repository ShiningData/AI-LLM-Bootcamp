from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
import json

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

# Tools that access persistent store (survives across conversations)
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up persistent user info from store."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    
    if user_info:
        data = user_info.value
        return f"💾 Stored User Info:\nID: {user_id}\nName: {data.get('name', 'N/A')}\nAge: {data.get('age', 'N/A')}\nEmail: {data.get('email', 'N/A')}"
    return f"❌ No stored info found for user {user_id}"

@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info to persistent store."""
    store = runtime.store
    user_data = {
        "name": name,
        "age": age,
        "email": email
    }
    
    store.put(("users",), user_id, user_data)
    return f"✅ Saved user info for {user_id} to persistent store"

@tool
def list_stored_users(runtime: ToolRuntime) -> str:
    """List all users in the persistent store."""
    store = runtime.store
    
    # Get all items in the users namespace
    try:
        items = list(store.search(("users",)))
        if not items:
            return "📭 No users stored yet"
            
        result = "📋 Stored Users:\n"
        for item in items:
            result += f"- {item.key}: {item.value.get('name', 'Unknown')}\n"
        return result
    except:
        return "📭 No users stored yet"

def main():
    print("💾 Memory Store Demo")
    print("="*30)
    print("This demonstrates PERSISTENT memory that survives across conversations!")
    print("\nTry these commands:")
    print("1. 'Save user abc123 named John age 30 email john@test.com'")
    print("2. 'Get user info for abc123'") 
    print("3. 'List all stored users'")
    print("4. Note: InMemoryStore only persists during this session (not across restarts)")
    print("\nType 'quit' to exit\n")
    
    # Create persistent store
    store = InMemoryStore()
    
    # Create agent with store
    agent = create_agent(
        model=model,
        tools=[get_user_info, save_user_info, list_stored_users],
        store=store,
        system_prompt="You help manage persistent user data. Use the tools to save and retrieve user information."
    )
    
    # Agent manages conversation history automatically in state
    state = {"messages": []}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        
        try:
            # Agent automatically manages the full conversation history
            result = agent.invoke(state)
            
            # Update state with agent's response (includes full conversation)
            state = result
            
            response = result["messages"][-1].content
            
            # Clean response
            if isinstance(response, list) and len(response) > 0:
                response = response[0].get('text', str(response))
                
            print(f"Agent: {response}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()