"""
Message summarization example with PostgreSQL memory.

This example shows how to use SummarizationMiddleware to automatically
summarize old messages when conversations become too long, preserving
important context while managing memory efficiently.

docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

# Initialize the main model for responding to users
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

# Initialize a smaller/faster model for generating summaries
summary_model = init_chat_model(
    model="gemini-2.5-flash-lite",  # Could use a smaller model like "gemini-1.5-flash"
    model_provider="google_genai", 
    max_tokens=300
)

@tool
def search_information(query: str) -> str:
    """Search for information about a topic."""
    return f"Found information about: {query} - This is detailed information that could be quite long and important to remember."

@tool
def save_user_data(data_type: str, value: str) -> str:
    """Save user data like preferences, personal information, etc."""
    return f"Saved {data_type}: {value}"

@tool
def get_recommendations(category: str) -> str:
    """Get recommendations for the user."""
    return f"Here are recommendations for {category}: Item1, Item2, Item3 - these are based on your preferences and past interactions."

def main():
    # PostgreSQL connection string
    DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db"
    
    try:
        # Create PostgreSQL checkpointer for persistent memory
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            print("📦 PostgreSQL memory initialized")
            
            # Create agent with summarization middleware
            agent = create_agent(
                model=model,  # Main model for responding to users
                tools=[search_information, save_user_data, get_recommendations],
                
                middleware=[
                    SummarizationMiddleware(
                        model=summary_model,              # Model used to create summaries
                        max_tokens_before_summary=1000,   # When to trigger summarization (lower for demo)
                        messages_to_keep=4,               # Keep last 4 raw messages after summarization
                        # The middleware will:
                        # 1. Count tokens in message history
                        # 2. If > max_tokens_before_summary, summarize older messages
                        # 3. Keep recent messages_to_keep messages as-is
                        # 4. Replace older messages with a summary message
                    )
                ],
                
                checkpointer=checkpointer,  # Persistent memory across sessions
                system_prompt="You are a helpful assistant with excellent memory. Remember user preferences, past conversations, and important details."
            )
            
            print("🤖 Agent with Message Summarization + PostgreSQL Memory")
            print("Long conversations are automatically summarized to preserve context")
            print("Type 'quit' to exit, 'count' to see message count, 'info' for current session info\n")
            
            # Configuration with thread ID for persistent conversations
            config = {"configurable": {"thread_id": "summary_session_1"}}
            
            # Simulate a longer conversation to demonstrate summarization
            test_messages = [
                "Hi, I'm Sarah and I'm a software engineer from San Francisco",
                "I'm interested in learning about machine learning and AI",
                "Can you save my preference that I like technical content?",
                "Search for information about neural networks for me",
                "I also enjoy reading about quantum computing",
                "Can you get me some recommendations for programming books?",
                "What's my name and what city am I from?",
                "What are my interests that we've discussed?",
                "Search for information about deep learning frameworks",
                "I'm planning to start a new project in Python",
                "Save that I prefer Python for data science projects",
                "What do you remember about my preferences and background?"
            ]
            
            print("🚀 Running extended conversation to demonstrate summarization:")
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
                
                # Show current message count and check for summarization
                message_count = len(result['messages'])
                print(f"💬 Messages in memory: {message_count}")
                
                # Check if any message is a summary (summarization occurred)
                has_summary = any("Summary of previous conversation" in str(msg) or 
                                "Previous conversation summary" in str(msg) or
                                hasattr(msg, 'content') and msg.content and 
                                ("summary" in str(msg.content).lower() or "summarize" in str(msg.content).lower())
                                for msg in result['messages'])
                
                if has_summary:
                    print("📝 ✨ Summarization detected! Old messages have been summarized.")
            
            print("\n" + "="*60)
            print("🎯 Interactive mode - Continue the conversation:")
            print("Commands: 'quit' (exit), 'count' (message count), 'info' (session info)")
            
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
                        
                        # Check for summaries in current messages
                        messages = current_state.values['messages']
                        summary_count = sum(1 for msg in messages if 
                                          hasattr(msg, 'content') and msg.content and 
                                          ("summary" in str(msg.content).lower() or 
                                           "Summary of previous" in str(msg.content)))
                        if summary_count > 0:
                            print(f"📝 Summary messages found: {summary_count}")
                    else:
                        print("💬 No messages in current thread")
                    continue
                elif user_input.lower() == 'info':
                    print("📊 Session Information:")
                    print("- Summarization triggers at 1000 tokens")
                    print("- Keeps last 4 messages as raw text")
                    print("- Older messages are condensed into summaries")
                    print("- All data persists in PostgreSQL database")
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
                    
                    # Show message count and summarization status
                    message_count = len(result['messages'])
                    print(f"💬 Messages in memory: {message_count}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("docker run -d --rm --name postgres-langgraph -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=langgraph_db -p 5432:5432 postgres:16")

if __name__ == "__main__":
    main()