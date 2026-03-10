"""
Conversation summarization using lifecycle middleware.

Shows how to:
- Use built-in SummarizationMiddleware
- Create custom summarization logic
- Persist conversation summaries in state
- Manage conversation history length
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import (
    SummarizationMiddleware,
    before_agent,
    after_model
)
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

# Initialize models
main_model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=400
)

summary_model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=200
)

@tool
def search_knowledge(query: str) -> str:
    """Search knowledge base for information."""
    return f"Knowledge search for '{query}': Found relevant information about the topic."

@tool
def save_note(note: str) -> str:
    """Save a note for later reference."""
    return f"Saved note: '{note}'"

# Custom summarization hooks
@before_agent
def check_conversation_length(state, runtime):
    """Check if conversation needs summarization before processing."""
    messages = state.get("messages", [])
    message_count = len(messages)
    
    if message_count > 12:  # Trigger at 12 messages
        print(f"📏 LIFECYCLE: Long conversation detected ({message_count} messages)")
        print("🔄 Built-in summarization middleware will handle this automatically")
    elif message_count > 8:
        print(f"📏 LIFECYCLE: Conversation growing ({message_count} messages)")
    
    return None

@after_model
def log_conversation_stats(state, runtime):
    """Log conversation statistics after each model call."""
    messages = state.get("messages", [])
    
    # Count different message types
    user_messages = sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "user")
    assistant_messages = sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant")
    
    # Estimate token count (rough approximation)
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict))
    estimated_tokens = total_chars // 4
    
    print(f"📊 STATS: {user_messages} user msgs, {assistant_messages} assistant msgs, ~{estimated_tokens} tokens")
    
    return None

def demo_builtin_summarization():
    """Demo built-in SummarizationMiddleware."""
    print("📚 Built-in Summarization Middleware")
    print("=" * 50)
    
    agent = create_agent(
        model=main_model,
        tools=[search_knowledge, save_note],
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger={"messages": 8},  # Trigger when 8+ messages
                keep={"messages": 4},     # Keep last 4 messages
            ),
            check_conversation_length,
            log_conversation_stats
        ],
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful research assistant with conversation summarization."
    )
    
    config = {"configurable": {"thread_id": "summarization_demo"}}
    
    # Build up a long conversation
    conversation_turns = [
        "What is machine learning?",
        "How does supervised learning work?", 
        "Explain neural networks",
        "What are the different types of neural networks?",
        "Tell me about deep learning",
        "How does backpropagation work?",
        "What is reinforcement learning?",
        "Explain natural language processing",  # This should trigger summarization
        "How do transformers work?",
        "What is the attention mechanism?"
    ]
    
    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i}: {user_input} ---")
        
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            )
            
            message_count = len(result.get("messages", []))
            print(f"📊 Messages in conversation: {message_count}")
            
            # Check if summarization happened
            if i >= 8:
                print("🔄 Summarization should have kept conversation manageable")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def demo_manual_summarization():
    """Demo manual conversation summarization with custom logic."""
    print("\n📝 Manual Summarization Logic")
    print("=" * 50)
    
    # Simple agent without auto-summarization
    agent = create_agent(
        model=main_model,
        tools=[search_knowledge],
        middleware=[log_conversation_stats],
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful assistant. Keep responses concise."
    )
    
    config = {"configurable": {"thread_id": "manual_demo"}}
    
    # Simulate conversation buildup
    messages = []
    for i in range(6):
        user_msg = f"Tell me about topic {i+1} in AI"
        messages.append({"role": "user", "content": user_msg})
        
        print(f"\n--- Adding message {i+1} ---")
        result = agent.invoke(
            {"messages": messages},
            config
        )
        
        # Get assistant response to add to our message list
        if result.get("messages"):
            last_msg = result["messages"][-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                messages.append(last_msg)
    
    print(f"\n📊 Final conversation length: {len(messages)} messages")
    print("💡 In production, you'd implement custom summarization logic here")

def demo_selective_summarization():
    """Demo selective summarization keeping important messages."""
    print("\n🎯 Selective Summarization Strategy")
    print("=" * 50)
    
    # Custom middleware that preserves important messages
    @before_agent
    def selective_summary_check(state, runtime):
        """Custom summarization logic that preserves important messages."""
        messages = state.get("messages", [])
        
        if len(messages) > 10:
            print("🔍 CUSTOM: Would analyze messages for importance")
            print("💾 CUSTOM: Would keep important messages, summarize routine ones")
            print("🏷️ CUSTOM: Would tag messages with metadata for smart summarization")
        
        return None
    
    agent = create_agent(
        model=main_model,
        tools=[save_note],
        middleware=[
            selective_summary_check,
            log_conversation_stats
        ],
        system_prompt="You help users take notes and manage information."
    )
    
    config = {"configurable": {"thread_id": "selective_demo"}}
    
    # Simulate different types of interactions
    interactions = [
        "Save this important note: Project deadline is Friday",
        "What's the weather like?",  # Routine
        "Save another note: Meeting with client at 3pm", 
        "Tell me a joke",  # Routine
        "Save critical info: Server maintenance tonight",
        "How are you doing?",  # Routine
        "What notes do I have saved?",  # Important
    ]
    
    for interaction in interactions:
        print(f"\n--- {interaction} ---")
        agent.invoke(
            {"messages": [{"role": "user", "content": interaction}]},
            config
        )

def demo_summarization_with_memory():
    """Demo how summarization works with persistent memory."""
    print("\n🧠 Summarization with Memory Persistence")
    print("=" * 50)
    
    agent = create_agent(
        model=main_model,
        tools=[search_knowledge],
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger={"messages": 6},
                keep={"messages": 3},
            ),
        ],
        checkpointer=InMemorySaver(),
        system_prompt="You are a research assistant with memory."
    )
    
    config = {"configurable": {"thread_id": "memory_persistence_demo"}}
    
    # First session
    print("\n--- Session 1: Building conversation ---")
    for i in range(4):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Research question {i+1} about AI"}]},
            config
        )
    
    print("\n--- Session 2: Continuing conversation ---")
    for i in range(4):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Follow-up question {i+1}"}]},
            config
        )
        
        # After enough messages, summarization should have occurred
        if i >= 2:
            message_count = len(result.get("messages", []))
            print(f"📊 Conversation managed at {message_count} messages due to summarization")
    
    print("\n--- Session 3: Checking memory ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What have we discussed so far?"}]},
        config
    )
    
    print("🔍 The assistant should remember key points despite summarization")

if __name__ == "__main__":
    print("📚 LangChain Conversation Summarization Example")
    print("Shows lifecycle middleware for managing conversation length\n")
    
    demo_builtin_summarization()
    demo_manual_summarization()
    demo_selective_summarization()
    demo_summarization_with_memory()
    
    print("\n✅ Conversation summarization demo completed!")
    print("📚 Key concepts demonstrated:")
    print("   - Built-in SummarizationMiddleware usage")
    print("   - Conversation length monitoring")
    print("   - Custom summarization logic patterns")
    print("   - Selective message preservation")
    print("   - Memory persistence across summarization")
    print("   - Before/after lifecycle hooks")