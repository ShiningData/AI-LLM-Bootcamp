"""
Message trimming example with PostgreSQL memory.

This example shows how to use middleware to trim message history
to keep conversations within context window limits while maintaining
persistent memory in PostgreSQL.

docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16
"""
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import before_model
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
def get_user_info(query: str) -> str:
    """Get information about the user."""
    return f"User info: {query}"

@tool
def save_note(note: str) -> str:
    """Save a user note."""
    return f"Note saved: {note}"

# Middleware to trim messages before model call
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # runtime required by decorator
    """Keep only recent messages to fit context window."""
    
    # Extract current conversation messages
    messages = state["messages"]
    
    # If conversation is short, no trimming needed
    if len(messages) <= 5:
        return None
    
    print(f"📝 Trimming messages: {len(messages)} -> keeping first + last 3")
    
    # Always keep the first message (usually system/initial)
    first_msg = messages[0]
    
    # Keep last 3 messages
    recent_messages = messages[-3:]
    
    # Create new trimmed message list
    new_messages = [first_msg] + recent_messages
    
    # Return instructions to update agent memory:
    # 1) Remove all existing messages
    # 2) Add back only trimmed messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # Clear all messages
            *new_messages                           # Add trimmed history
        ]
    }

def main():
    # PostgreSQL connection string
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        # Create PostgreSQL checkpointer for persistent memory
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized")
            
            # Create agent with message trimming middleware
            agent = create_agent(
                model=model,
                tools=[get_user_info, save_note],
                middleware=[trim_messages],  # Add trimming middleware
                checkpointer=checkpointer,   # Persistent memory
                system_prompt="You are a helpful assistant. Remember user conversations but keep responses concise."
            )
            
            print("🤖 Agent with Message Trimming + PostgreSQL Memory")
            print("Message history will be automatically trimmed when it gets too long")
            print("Type 'quit' to exit, 'count' to see message count\n")
            
            # Configuration with thread ID for persistent conversations
            config = {"configurable": {"thread_id": "trim_session_1"}}
            
            # Simulate a long conversation to demonstrate trimming
            test_messages = [
                "Hi, my name is Alice",
                "I work as a software engineer",
                "I love programming in Python",
                "Can you save a note that I prefer dark mode?",
                "What programming languages do you recommend?",
                "Tell me about machine learning",
                "Do you remember my name?"
            ]
            
            print("🚀 Running test conversation to demonstrate message trimming:")
            for i, msg in enumerate(test_messages, 1):
                print(f"\n--- Message {i}: {msg} ---")
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": msg}]},
                    config
                )
                
                # Show agent response
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
                
                # Show current message count
                print(f"💬 Current message count: {len(result['messages'])}")
            
            print("\n" + "="*60)
            print("🎯 Interactive mode - Continue the conversation:")
            print("Type 'quit' to exit, 'count' to see current message count")
            
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
                
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config
                )
                
                # Display agent response
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