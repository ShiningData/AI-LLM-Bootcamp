"""
docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgre
s -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16
"""
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def search_info(query: str) -> str:
    """Search for information about a topic."""
    return f"Found detailed information about: {query}"

@tool
def save_user_preference(preference_type: str, value: str) -> str:
    """Save a user preference."""
    return f"Saved preference: {preference_type} = {value}"

class CustomAgentState(AgentState):
    """Extended agent state with custom fields for user data."""
    user_id: str = "default_user"
    preferences: dict = {}
    session_count: int = 0

def main():
    # PostgreSQL connection string
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        # Create PostgreSQL checkpointer for persistent memory
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # Setup database tables automatically
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized successfully")
            
            # Create agent with PostgreSQL-backed memory
            agent = create_agent(
                model=model,
                tools=[search_info, save_user_preference],
                state_schema=CustomAgentState,
                checkpointer=checkpointer,  # Provides persistent memory to AI assistant app
                system_prompt="You are a helpful assistant with persistent memory. Remember user preferences and conversation history across sessions."
            )
            
            print("🤖 Agent with PostgreSQL Short-Term Memory")
            print("Your conversation history is saved in PostgreSQL database")
            print("Type 'quit' to exit, 'history' to see conversation count")
            print("Try asking about previous conversations after restarting!\n")
            
            # Configuration with thread ID for persistent conversations
            config = {"configurable": {"thread_id": "user_session_1"}}
            
            while True:
                user_input = input("You: ")
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'history':
                    print("📊 This conversation persists in PostgreSQL database")
                    print("Restart the program and use the same thread_id to continue!")
                    continue
                
                # Invoke agent with custom state that matches CustomAgentState schema
                # The agent uses these fields from our extended AgentState:
                # - messages: conversation history (inherited from base AgentState)
                # - user_id: custom field for user identification
                # - preferences: custom field for user settings
                # - session_count: custom field for tracking interactions
                result = agent.invoke(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                        "user_id": "user_123",
                        "preferences": {"language": "english", "style": "friendly"},
                        "session_count": 1
                    },
                    config
                )
                
                # Extract and display the agent's response
                # The result contains the updated state with all messages
                if result and "messages" in result and result["messages"]:
                    # Get the most recent message (agent's response)
                    agent_message = result["messages"][-1]
                    
                    # Check if message has content attribute (normal case)
                    if hasattr(agent_message, 'content'):
                        content = agent_message.content
                        
                        # Handle complex content format: [{'type': 'text', 'text': '...', 'extras': {...}}]
                        # Some models return content as a list with metadata we don't want to show
                        if isinstance(content, list) and len(content) > 0:
                            # Look for 'text' field in the first content item
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                text_content = content[0]['text']  # Extract just the text
                            else:
                                text_content = str(content[0])  # Fallback to string conversion
                        else:
                            # Simple content format (just a string)
                            text_content = str(content)
                        
                        print(f"Agent: {text_content}\n")
                    else:
                        # Fallback if message structure is unexpected
                        print(f"Agent: {agent_message}\n")
                
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running with:")
        print("- Host: localhost")
        print("- Port: 5432") 
        print("- Database: langgraph_db")
        print("- Username/Password: postgres/postgres")
        print("\nTo start PostgreSQL with Docker:")
        print("docker run -d --name postgres-langgraph -e POSTGRES_DB=langgraph_db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15")

if __name__ == "__main__":
    main()