"""
Message deletion example with PostgreSQL memory.

This example shows how to use middleware to delete specific messages
from conversation history while maintaining persistent memory in PostgreSQL.

docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16
"""
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 22°C"

@tool
def set_reminder(reminder: str) -> str:
    """Set a reminder for the user."""
    return f"Reminder set: {reminder}"

# Middleware to delete old messages after model response
@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # runtime required by decorator
    """Remove older messages to keep conversation manageable."""
    
    # Get current message history from agent state
    messages = state["messages"]
    
    # If we have more than 4 messages, delete the earliest 2
    if len(messages) > 4:
        print(f"🗑️  Deleting old messages: {len(messages)} -> removing oldest 2")
        
        # Create RemoveMessage commands for the oldest messages
        # messages[:2] gets the first 2 messages, then we create RemoveMessage for each
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    
    # If not enough messages to warrant deletion, do nothing
    return None

# Alternative function to delete ALL messages (useful for clearing conversation)
def delete_all_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # runtime required by decorator
    """Remove all messages from conversation history."""
    messages = state["messages"]
    
    if len(messages) > 0:
        print("🗑️  Clearing ALL conversation history")
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
    
    return None

def main():
    # PostgreSQL connection string
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        # Create PostgreSQL checkpointer for persistent memory
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized")
            
            # Create agent with message deletion middleware
            agent = create_agent(
                model=model,
                tools=[get_weather, set_reminder],
                middleware=[delete_old_messages],  # Add deletion middleware (runs AFTER model)
                checkpointer=checkpointer,         # Persistent memory
                system_prompt="You are a helpful assistant. Keep responses concise."
            )
            
            print("🤖 Agent with Message Deletion + PostgreSQL Memory")
            print("Old messages are automatically deleted after each response")
            print("Type 'quit' to exit, 'count' to see message count, 'clear' to delete all messages\n")
            
            # Configuration with thread ID for persistent conversations
            config = {"configurable": {"thread_id": "delete_session_1"}}
            
            # Simulate conversation to demonstrate message deletion
            test_messages = [
                "Hi, I'm John from New York",
                "What's the weather like in my city?",
                "Can you set a reminder to call mom tomorrow?",
                "Tell me a joke",
                "What's my name again?",  # This should test if name is remembered after deletions
                "What city am I from?"   # This should test if location is remembered
            ]
            
            print("🚀 Running test conversation to demonstrate message deletion:")
            for i, msg in enumerate(test_messages, 1):
                print(f"\n--- Message {i}: {msg} ---")
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": msg}]},
                    config
                )
                
                # Extract and display the agent's response from the result
                if result and "messages" in result:
                    # Get the last message (agent's response)
                    agent_message = result["messages"][-1]
                    
                    # Check if message has content (standard message format)
                    if hasattr(agent_message, 'content'):
                        content = agent_message.content
                        
                        # Handle different content formats:
                        # Some models return: [{'type': 'text', 'text': 'actual response', 'extras': {...}}]
                        # Others return: simple string content
                        if isinstance(content, list) and len(content) > 0:
                            # Complex format - extract text from dictionary or convert to string
                            text_content = content[0].get('text', str(content[0])) if isinstance(content[0], dict) else str(content[0])
                        else:
                            # Simple format - just convert to string
                            text_content = str(content)
                        print(f"Agent: {text_content}")
                
                # Show current message count after deletion middleware runs
                print(f"💬 Messages remaining after deletion: {len(result['messages'])}")
            
            print("\n" + "="*60)
            print("🎯 Interactive mode - Continue the conversation:")
            print("Commands: 'quit' (exit), 'count' (message count), 'clear' (delete all)")
            
            while True:
                user_input = input("\nYou: ")
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'count':
                    # Get current state to check message count
                    current_state = checkpointer.get(config)
                    if current_state and current_state.values.get('messages'):
                        msg_count = len(current_state.values['messages'])
                        print(f"💬 Current message count: {msg_count}")
                    else:
                        print("💬 No messages in current thread")
                    continue
                elif user_input.lower() == 'clear':
                    # Manually clear all messages by invoking with clear command
                    print("🗑️  Clearing all messages...")
                    result = agent.invoke(
                        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]},
                        config
                    )
                    print("✅ All messages cleared")
                    continue
                
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config
                )
                
                # Extract and display the agent's response from the result
                if result and "messages" in result:
                    # Get the last message (agent's response)
                    agent_message = result["messages"][-1]
                    
                    # Check if message has content (standard message format)
                    if hasattr(agent_message, 'content'):
                        content = agent_message.content
                        
                        # Handle different content formats:
                        # Some models return: [{'type': 'text', 'text': 'actual response', 'extras': {...}}]
                        # Others return: simple string content
                        if isinstance(content, list) and len(content) > 0:
                            # Complex format - extract text from dictionary or convert to string
                            text_content = content[0].get('text', str(content[0])) if isinstance(content[0], dict) else str(content[0])
                        else:
                            # Simple format - just convert to string
                            text_content = str(content)
                        print(f"Agent: {text_content}")
                    
                    # Show message count after each interaction
                    print(f"💬 Messages in memory: {len(result['messages'])}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16")

if __name__ == "__main__":
    main()