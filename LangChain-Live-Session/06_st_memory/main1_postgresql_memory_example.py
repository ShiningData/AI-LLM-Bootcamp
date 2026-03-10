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
            checkpointer.setup()
            # Create agent with PostgreSQL-backed memory
            agent = create_agent(
                model=model,
                tools=[search_info, save_user_preference],
                state_schema=CustomAgentState,
                checkpointer=checkpointer,  # Provides persistent memory to AI assistant app
                system_prompt="You are a helpful assistant with persistent memory. Remember user preferences and conversation history across sessions."
            )

            config = {"configurable": {"thread_id": "user_session_1"}}

            while True:
                user_input = input("You: ")
                
                if user_input.lower() == 'quit':
                    break
                
                result = agent.invoke(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                        "user_id": "user_123",
                        "preferences": {"language": "english", "style": "friendly"},
                        "session_count": 1
                    },
                    config
                )

                print(result["messages"][-1])

    except Exception as e:
         print(f"❌ Error connecting to PostgreSQL: {e}")

if __name__=='__main__':
    main()