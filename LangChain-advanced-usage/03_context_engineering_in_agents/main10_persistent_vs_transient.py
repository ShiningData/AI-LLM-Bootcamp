"""
Persistent vs transient context updates in lifecycle middleware.

Shows the difference between:
- Transient updates: Temporary modifications for single calls
- Persistent updates: Permanent changes to state/store
- When to use each approach
- How updates propagate through conversation turns
"""
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest, 
    ModelResponse,
    before_agent,
    after_agent
)
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@dataclass
class UserContext:
    user_id: str
    preferences: dict

@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 22°C"

@tool
def set_preference(key: str, value: str) -> str:
    """Set a user preference."""
    return f"Set preference: {key} = {value}"

# 1. TRANSIENT updates (temporary, one-time modifications)
@wrap_model_call
def transient_context_injection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Transiently inject context for this specific model call only."""
    
    user_prefs = request.runtime.context.preferences
    
    # Build temporary context message
    if user_prefs:
        context_msg = f"""[Temporary context for this response only]
Your current preferences: {user_prefs}
Use these preferences to customize your response style."""
        
        # Temporarily add context message (NOT saved to state)
        modified_messages = [
            *request.messages,
            {"role": "user", "content": context_msg}
        ]
        
        print("🔄 TRANSIENT: Added temporary context for this model call")
        request = request.override(messages=modified_messages)
    
    return handler(request)

@wrap_model_call  
def transient_model_adjustment(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Transiently adjust model behavior based on message count."""
    
    message_count = len(request.messages)
    
    if message_count > 8:
        # Temporarily instruct model to be more concise
        concise_instruction = "Be extra concise in your response due to conversation length."
        
        modified_messages = [
            *request.messages,
            {"role": "user", "content": concise_instruction}
        ]
        
        print("🔄 TRANSIENT: Temporarily requesting concise responses")
        request = request.override(messages=modified_messages)
    
    return handler(request)

# 2. PERSISTENT updates (permanent changes to state/store)
@before_agent
def persistent_conversation_tracking(state, runtime):
    """Persistently track conversation metadata in state."""
    
    # Get current metadata or initialize
    conversation_meta = state.get("conversation_metadata", {
        "turn_count": 0,
        "topics_discussed": [],
        "user_preferences_mentioned": []
    })
    
    # Increment turn count
    conversation_meta["turn_count"] += 1
    
    # Analyze current message for topics
    messages = state.get("messages", [])
    if messages:
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        if last_user_msg:
            # Simple topic detection
            if "weather" in last_user_msg:
                if "weather" not in conversation_meta["topics_discussed"]:
                    conversation_meta["topics_discussed"].append("weather")
            if "preference" in last_user_msg:
                if "preferences" not in conversation_meta["topics_discussed"]:
                    conversation_meta["topics_discussed"].append("preferences")
    
    print(f"💾 PERSISTENT: Updated conversation metadata (turn {conversation_meta['turn_count']})")
    
    # Return persistent update to state
    return {"conversation_metadata": conversation_meta}

@after_agent
def persistent_user_behavior_tracking(state, runtime):
    """Persistently track user behavior patterns in store."""
    
    user_id = runtime.context.user_id
    store = runtime.store
    
    if store:
        try:
            # Get existing behavior data
            behavior_data = store.get(("user_behavior",), user_id)
            if behavior_data:
                behavior = behavior_data.value
            else:
                behavior = {
                    "total_interactions": 0,
                    "avg_message_length": 0,
                    "preferred_topics": {},
                    "session_count": 0
                }
            
            # Update behavior tracking
            behavior["total_interactions"] += 1
            
            # Analyze current session
            messages = state.get("messages", [])
            user_messages = [
                msg for msg in messages 
                if isinstance(msg, dict) and msg.get("role") == "user"
            ]
            
            if user_messages:
                # Calculate average message length
                total_length = sum(len(msg.get("content", "")) for msg in user_messages)
                current_avg = total_length / len(user_messages)
                
                # Update running average
                prev_avg = behavior["avg_message_length"]
                prev_count = behavior["total_interactions"] - 1
                behavior["avg_message_length"] = (
                    (prev_avg * prev_count + current_avg) / behavior["total_interactions"]
                )
            
            # Track topics from metadata
            conversation_meta = state.get("conversation_metadata", {})
            topics = conversation_meta.get("topics_discussed", [])
            for topic in topics:
                behavior["preferred_topics"][topic] = behavior["preferred_topics"].get(topic, 0) + 1
            
            # Save back to store
            store.put(("user_behavior",), user_id, behavior)
            print(f"💾 PERSISTENT: Updated user behavior tracking in store")
            
        except Exception as e:
            print(f"❌ Error updating behavior tracking: {e}")
    
    return None

# 3. Mixed approach - transient with conditional persistence
@wrap_model_call
def adaptive_response_formatting(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Adaptively format responses with mixed transient/persistent updates."""
    
    user_prefs = request.runtime.context.preferences
    conversation_meta = request.state.get("conversation_metadata", {})
    turn_count = conversation_meta.get("turn_count", 0)
    
    # TRANSIENT: Add formatting instruction for this call only
    format_instruction = ""
    
    if user_prefs.get("response_style") == "bullet_points":
        format_instruction = "Format your response using bullet points."
    elif user_prefs.get("response_style") == "numbered_list":
        format_instruction = "Format your response as a numbered list."
    elif turn_count > 5:
        format_instruction = "Keep your response brief and to the point."
    
    if format_instruction:
        modified_messages = [
            *request.messages,
            {"role": "user", "content": f"[Formatting: {format_instruction}]"}
        ]
        
        print(f"🔄 TRANSIENT: Applied format instruction: {format_instruction}")
        request = request.override(messages=modified_messages)
    
    # Call the model with transient modifications
    response = handler(request)
    
    # PERSISTENT: Update formatting preferences based on user satisfaction
    # (In a real system, you'd track user feedback)
    if turn_count > 3 and format_instruction:
        print("💾 PERSISTENT: Would update user formatting preferences based on engagement")
    
    return response

def demo_transient_updates():
    """Demo transient updates that don't persist."""
    print("🔄 Transient Context Updates")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            transient_context_injection,
            transient_model_adjustment
        ],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful assistant."
    )
    
    config = {"configurable": {"thread_id": "transient_demo"}}
    context = UserContext(
        user_id="user123", 
        preferences={"tone": "formal", "detail": "high"}
    )
    
    # First interaction - should have transient context
    print("\n--- Turn 1: Transient context injection ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather like in Paris?"}]},
        context=context,
        config=config
    )
    
    # Build up conversation to trigger conciseness
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"Question {i+1}"})
        messages.append({"role": "assistant", "content": f"Answer {i+1}"})
    
    print("\n--- Turn 2: Long conversation (transient conciseness) ---")
    agent.invoke(
        {
            "messages": messages + [{"role": "user", "content": "What's the weather in Tokyo?"}]
        },
        context=context,
        config=config
    )

def demo_persistent_updates():
    """Demo persistent updates that maintain across turns."""
    print("\n💾 Persistent Context Updates")
    print("=" * 50)
    
    store = InMemoryStore()
    
    agent = create_agent(
        model=model,
        tools=[set_preference],
        middleware=[
            persistent_conversation_tracking,
            persistent_user_behavior_tracking
        ],
        context_schema=UserContext,
        store=store,
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful assistant with memory."
    )
    
    config = {"configurable": {"thread_id": "persistent_demo"}}
    context = UserContext(user_id="persistent_user", preferences={})
    
    # Multiple interactions to build persistent state
    interactions = [
        "What's the weather like today?",
        "Can you help me set a preference for notifications?",
        "Tell me about the weather forecast",
        "I'd like to update my preferences again"
    ]
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n--- Turn {i}: {interaction} ---")
        agent.invoke(
            {"messages": [{"role": "user", "content": interaction}]},
            context=context,
            config=config
        )
    
    # Check what was persisted in store
    print("\n--- Checking persistent store data ---")
    try:
        behavior_data = store.get(("user_behavior",), "persistent_user")
        if behavior_data:
            print(f"Stored behavior data: {behavior_data.value}")
        else:
            print("No behavior data found in store")
    except Exception as e:
        print(f"Error reading store: {e}")

def demo_mixed_approach():
    """Demo mixing transient and persistent updates."""
    print("\n🔄💾 Mixed Transient/Persistent Updates")
    print("=" * 50)
    
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            persistent_conversation_tracking,  # Persistent
            adaptive_response_formatting       # Mixed
        ],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        system_prompt="You are an adaptive assistant."
    )
    
    config = {"configurable": {"thread_id": "mixed_demo"}}
    
    # Test different preference styles
    preferences_tests = [
        {"response_style": "bullet_points"},
        {"response_style": "numbered_list"},
        {"response_style": "paragraph"}
    ]
    
    for i, prefs in enumerate(preferences_tests, 1):
        print(f"\n--- Test {i}: {prefs} ---")
        context = UserContext(user_id=f"mixed_user_{i}", preferences=prefs)
        
        agent.invoke(
            {"messages": [{"role": "user", "content": "Tell me about renewable energy sources"}]},
            context=context,
            config=config
        )

def demo_persistence_across_sessions():
    """Demo how persistent updates survive across different sessions."""
    print("\n🔄 Persistence Across Sessions")
    print("=" * 50)
    
    store = InMemoryStore()
    
    # Session 1: Build up some persistent state
    print("\n--- Session 1: Building persistent state ---")
    agent1 = create_agent(
        model=model,
        tools=[set_preference],
        middleware=[persistent_conversation_tracking, persistent_user_behavior_tracking],
        context_schema=UserContext,
        store=store,
        checkpointer=InMemorySaver(),
        system_prompt="Session 1 assistant."
    )
    
    config1 = {"configurable": {"thread_id": "session_1"}}
    context = UserContext(user_id="cross_session_user", preferences={})
    
    agent1.invoke(
        {"messages": [{"role": "user", "content": "I want to talk about weather and preferences"}]},
        context=context,
        config=config1
    )
    
    agent1.invoke(
        {"messages": [{"role": "user", "content": "Let me set some preferences for my account"}]},
        context=context,
        config=config1
    )
    
    # Session 2: Check if persistent state is available
    print("\n--- Session 2: Checking persistent state ---")
    agent2 = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[persistent_user_behavior_tracking],
        context_schema=UserContext,
        store=store,
        checkpointer=InMemorySaver(),
        system_prompt="Session 2 assistant."
    )
    
    config2 = {"configurable": {"thread_id": "session_2"}}
    
    agent2.invoke(
        {"messages": [{"role": "user", "content": "What do you know about my previous interactions?"}]},
        context=context,
        config=config2
    )
    
    # Check final persistent state
    print("\n--- Final persistent state check ---")
    try:
        behavior_data = store.get(("user_behavior",), "cross_session_user")
        if behavior_data:
            print(f"Persistent behavior data: {behavior_data.value}")
        else:
            print("No persistent data found")
    except Exception as e:
        print(f"Error checking persistent state: {e}")

if __name__ == "__main__":
    print("🔄💾 LangChain Persistent vs Transient Updates Example")
    print("Shows the difference between temporary and permanent context modifications\n")
    
    demo_transient_updates()
    demo_persistent_updates()
    demo_mixed_approach()
    demo_persistence_across_sessions()
    
    print("\n✅ Persistent vs transient demo completed!")
    print("🔄💾 Key concepts demonstrated:")
    print("   - Transient updates with @wrap_model_call")
    print("   - Persistent updates with before/after hooks")
    print("   - State vs Store persistence")
    print("   - Mixed transient/persistent strategies")
    print("   - Cross-session persistence")
    print("   - When to use each approach")