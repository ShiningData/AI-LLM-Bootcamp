"""
Streaming example with LangChain agents.

In this example you will see:
- How to stream agent progress with stream_mode="updates"
- How to stream LLM tokens with stream_mode="messages" 
- How to emit custom updates from tools using get_stream_writer
- How to stream multiple modes simultaneously
- How streaming improves user experience with real-time feedback
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.config import get_stream_writer
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # Get stream writer for custom updates
    writer = get_stream_writer()
    
    # Emit custom progress updates
    writer(f"🔍 Looking up weather data for {city}...")
    writer(f"📡 Connecting to weather API...")
    writer(f"✅ Successfully retrieved data for {city}")
    
    return f"The weather in {city} is sunny and 22°C!"

@tool
def get_time(timezone: str) -> str:
    """Get current time for a timezone."""
    writer = get_stream_writer()
    
    writer(f"⏰ Checking time for timezone: {timezone}")
    writer(f"🌍 Converting to local time...")
    
    return f"Current time in {timezone} is 14:30 UTC"

# Create agent with tools
agent = create_agent(
    model=model,
    tools=[get_weather, get_time],
    system_prompt="You are a helpful assistant. Use tools when needed and provide clear responses."
)

def demo_agent_progress_streaming():
    """Demo: Stream agent progress (node-level updates)."""
    print("🚀 DEMO 1: Streaming Agent Progress (updates mode)")
    print("=" * 60)
    
    # Stream agent progress - shows each step the agent takes
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
        stream_mode="updates"  # Shows progress after each agent step
    ):
        # Each chunk contains updates from different nodes (model, tools, etc.)
        for step, data in chunk.items():
            print(f"📍 Step: {step}")
            # Show the last message from this step
            if "messages" in data and data["messages"]:
                last_msg = data["messages"][-1]
                if hasattr(last_msg, 'content'):
                    print(f"   Content: {last_msg.content}")
                print()

def demo_token_streaming():
    """Demo: Stream LLM tokens as they are generated."""
    print("\n🚀 DEMO 2: Streaming LLM Tokens (messages mode)")
    print("=" * 60)
    
    # Stream individual tokens/messages as they're generated
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "Tell me about the weather in Tokyo"}]},
        stream_mode="messages"  # Stream tokens and messages
    ):
        # Show which node produced this token
        node = metadata.get('langgraph_node', 'unknown')
        print(f"🔗 Node: {node}")
        
        # Show the token content
        if hasattr(token, 'content'):
            print(f"   Token: {token.content}")
        print()

def demo_custom_streaming():
    """Demo: Stream custom updates from tools."""
    print("\n🚀 DEMO 3: Streaming Custom Updates (custom mode)")
    print("=" * 60)
    
    # Stream only custom updates emitted by tools using get_stream_writer
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What time is it in New York?"}]},
        stream_mode="custom"  # Only show custom updates from tools
    ):
        # Each chunk is whatever was passed to writer()
        print(f"📢 Custom Update: {chunk}")

def demo_multiple_modes():
    """Demo: Stream multiple modes simultaneously."""
    print("\n🚀 DEMO 4: Streaming Multiple Modes (updates + custom)")
    print("=" * 60)
    
    # Stream both agent progress AND custom updates
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Check weather in London and time in GMT"}]},
        stream_mode=["updates", "custom"]  # Multiple modes as a list
    ):
        print(f"🏷️  Stream Mode: {stream_mode}")
        
        if stream_mode == "updates":
            # Handle agent progress updates
            for step, data in chunk.items():
                print(f"   📍 Agent Step: {step}")
                if "messages" in data and data["messages"]:
                    last_msg = data["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        content = str(last_msg.content)[:100]  # Truncate for display
                        print(f"      Content: {content}...")
        
        elif stream_mode == "custom":
            # Handle custom tool updates
            print(f"   📢 Tool Update: {chunk}")
        
        print()

if __name__ == "__main__":
    print("🌟 LangChain Streaming Examples")
    print("This demo shows different ways to stream agent responses\n")
    
    # Run all streaming demos
    demo_agent_progress_streaming()
    demo_token_streaming() 
    demo_custom_streaming()
    demo_multiple_modes()
    
    print("\n✅ All streaming demos completed!")
    print("💡 Streaming improves UX by showing progress in real-time")