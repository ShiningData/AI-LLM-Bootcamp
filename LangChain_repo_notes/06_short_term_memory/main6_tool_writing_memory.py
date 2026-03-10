"""
Tool writing memory example with PostgreSQL.

Shows how tools can modify agent state using Command to update memory.
Tools can save information that persists across interactions.
"""
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

# Custom state for storing user data
class UserState(AgentState):
    user_name: str = "Unknown"
    user_preferences: dict = {}
    notes: list = []

# Custom context (per-request data, not persisted)
class UserContext(BaseModel):
    user_id: str

# Tool that updates user information in memory
@tool
def update_user_profile(
    name: str,
    runtime: ToolRuntime[UserContext, UserState]
) -> Command:
    """Update user profile information in memory."""
    
    # Return Command to update the agent's state
    return Command(update={
        "user_name": name,  # Update the user_name field
        "messages": [
            ToolMessage(
                f"Updated user name to: {name}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# Tool that saves user preferences to memory
@tool
def save_preference(
    preference_type: str,
    value: str,
    runtime: ToolRuntime[UserContext, UserState]
) -> Command:
    """Save a user preference to memory."""
    
    # Get current preferences and add new one
    current_prefs = runtime.state.get("user_preferences", {})
    current_prefs[preference_type] = value
    
    return Command(update={
        "user_preferences": current_prefs,
        "messages": [
            ToolMessage(
                f"Saved preference: {preference_type} = {value}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# Tool that adds notes to memory
@tool
def add_note(
    note: str,
    runtime: ToolRuntime[UserContext, UserState]
) -> Command:
    """Add a note to user's memory."""
    
    # Get current notes and add new one
    current_notes = runtime.state.get("notes", [])
    current_notes.append(note)
    
    return Command(update={
        "notes": current_notes,
        "messages": [
            ToolMessage(
                f"Added note: {note}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# Tool that reads current memory state
@tool
def get_my_info(
    runtime: ToolRuntime[UserContext, UserState]
) -> str:
    """Get current user information from memory."""
    
    name = runtime.state["user_name"]
    prefs = runtime.state["user_preferences"]
    notes = runtime.state["notes"]
    
    info = f"Your Information:\n"
    info += f"- Name: {name}\n"
    info += f"- Preferences: {prefs}\n"
    info += f"- Notes: {notes}"
    
    return info

def main():
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized")
            
            # Create agent with memory-writing tools
            agent = create_agent(
                model=model,
                tools=[update_user_profile, save_preference, add_note, get_my_info],
                state_schema=UserState,
                context_schema=UserContext,
                checkpointer=checkpointer,
                system_prompt="You help users manage their profile and preferences. Use tools to save and retrieve information."
            )
            
            config = {"configurable": {"thread_id": "write_session"}}
            context = UserContext(user_id="user_123")
            
            print("🤖 Tool Writing Memory Example")
            print("Tools can update agent memory state")
            print("Try: 'set my name to John', 'save preference theme dark', 'add note I like Python'\n")
            
            while True:
                try:
                    user_input = input("You: ")
                    
                    if user_input.lower() == 'quit':
                        break
                    
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config,
                        context=context
                    )
                    
                    # Display response - extract clean text
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                        # Handle complex content format
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                text_content = content[0]['text']
                            else:
                                text_content = str(content[0])
                        else:
                            text_content = str(content)
                        print(f"Agent: {text_content}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure PostgreSQL is running:")
        print("docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16")

if __name__ == "__main__":
    main()