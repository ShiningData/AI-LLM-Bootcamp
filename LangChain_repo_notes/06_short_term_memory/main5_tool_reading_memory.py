"""
Simple tool reading memory example - FIXED VERSION
Shows how tools can access agent state using ToolRuntime.
"""
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

# Custom state with user info
class UserState(AgentState):
    user_name: str = "Unknown"
    user_id: str = "unknown"
    preferences: dict = {}

# Tool that reads user info from memory
@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """Get user profile information from memory."""
    name = runtime.state["user_name"]
    user_id = runtime.state["user_id"]
    prefs = runtime.state["preferences"]
    
    return f"User Profile:\n- Name: {name}\n- ID: {user_id}\n- Preferences: {prefs}"

# Tool that counts messages
@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Get the number of messages in conversation."""
    messages = runtime.state["messages"]
    return f"Total messages in conversation: {len(messages)}"

def main():
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized")
            
            # Create agent with memory-reading tools
            agent = create_agent(
                model=model,
                tools=[get_user_profile, get_message_count],
                state_schema=UserState,
                checkpointer=checkpointer,
                system_prompt="You are helpful. Use tools to get information when asked."
            )
            
            config = {"configurable": {"thread_id": "simple_session"}}
            
            print("🤖 Simple Tool Memory Example")
            print("Commands: 'profile', 'count', 'quit'\n")
            
            # Initialize with user data
            agent.invoke({
                "messages": [{"role": "user", "content": "Hello"}],
                "user_name": "Alice",
                "user_id": "alice_123",
                "preferences": {"theme": "dark", "language": "python"}
            }, config)
            
            while True:
                try:
                    user_input = input("You: ")
                    
                    if user_input.lower() == 'quit':
                        break
                    
                    result = agent.invoke({
                        "messages": [{"role": "user", "content": user_input}]
                    }, config)
                    
                    # Simple response extraction
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        print(f"Agent: {last_msg.content}")
                    else:
                        print(f"Agent: {last_msg}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()