"""
Before and after model middleware example.

In this example you will see:
- How to use @before_model middleware to trim messages before LLM calls
- How to use @after_model middleware to validate responses after LLM calls
- How middleware can modify agent state by adding/removing messages
- How memory management works with trimming and validation
"""
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Initialize the model first
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=500
)

@tool
def get_info(query: str) -> str:
    """Get information about a topic."""
    return f"Information about: {query}"

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit within context window."""
    messages = state["messages"]
    
    if len(messages) <= 3:
        return None
    
    print(f"🔧 Trimming messages: {len(messages)} -> keeping first + last 2")
    
    first_msg = messages[0]
    recent_messages = messages[-2:]
    new_messages = [first_msg] + recent_messages
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove model responses that contain sensitive words."""
    STOP_WORDS = ["password", "secret", "confidential"]
    
    last_message = state["messages"][-1]
    
    # Handle different content formats
    content = last_message.content
    if isinstance(content, list) and len(content) > 0:
        # Extract text from complex content format
        text_content = content[0].get('text', str(content[0])) if isinstance(content[0], dict) else str(content[0])
    else:
        text_content = str(content)
    
    if any(word in text_content.lower() for word in STOP_WORDS):
        print(f"⚠️  Removing unsafe response containing sensitive words")
        return {
            "messages": [
                RemoveMessage(id=last_message.id)
            ]
        }
    
    return None

agent = create_agent(
    model=model,
    tools=[get_info],
    middleware=[trim_messages, validate_response],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant. Keep responses concise."
)

config = {"configurable": {"thread_id": "before_after_example"}}

print("🚀 Testing before_model and after_model middleware:")
print("- Before: Trims long message history")
print("- After: Validates and removes unsafe responses\n")

# Test conversation to demonstrate trimming
test_messages = [
    "Hi, my name is Alice",
    "I work as a software engineer", 
    "Tell me about Python programming",
    "What's my name again?",
    "Can you help me with my password reset?"  # This might trigger validation
]

for i, msg in enumerate(test_messages, 1):
    print(f"--- Message {i}: {msg} ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config
    )
    
    # Ensure we have valid results with messages
    if result and "messages" in result and result["messages"]:
        # Get the last message from the conversation
        agent_message = result["messages"][-1]
        # Check if the last message is from the agent (not user)
        # This prevents displaying user's question when agent response was filtered
        if hasattr(agent_message, 'role') and agent_message.role == 'assistant':
            # Check if message has content attribute
            if hasattr(agent_message, 'content'):
                # Extract the content from the message
                content = agent_message.content
                # Handle different content formats that models can return
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from complex list format: [{'text': '...'}] or [string]
                    text_content = content[0].get('text', str(content[0])) if isinstance(content[0], dict) else str(content[0])
                else:
                    # Handle simple string format
                    text_content = str(content)
                # Display the agent's response
                print(f"Agent: {text_content}")
        else:
            print("Agent: [Response was filtered/removed]")
        print(f"💬 Messages in memory: {len(result['messages'])}\n")