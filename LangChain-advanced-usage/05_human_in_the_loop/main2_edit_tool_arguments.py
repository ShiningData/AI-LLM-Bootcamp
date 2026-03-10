"""
Human-in-the-Loop with tool argument editing capabilities.

Shows how to:
- Allow humans to edit tool arguments before execution
- Handle edit decisions with modified parameters
- Implement different policies for different tool sensitivity levels
- Demonstrate the complete edit workflow
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai", 
    max_tokens=300
)

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file (editable operation)."""
    # Simulate file writing
    return f"File written to {filepath} with {len(content)} characters"

@tool
def execute_sql_query(query: str) -> str:
    """Execute a SQL query (approval only, no editing for security)."""
    # Simulate SQL execution
    return f"SQL query executed: {query[:50]}..."

@tool
def create_user_account(username: str, email: str, role: str = "user") -> str:
    """Create a new user account (fully editable)."""
    return f"User account created: {username} ({email}) with role '{role}'"

def extract_clean_content(message):
    """Extract clean text content from message, avoiding signature clutter."""
    if hasattr(message, 'content'):
        if isinstance(message.content, list) and len(message.content) > 0:
            return message.content[0].get('text', str(message.content))
        else:
            return str(message.content)
    else:
        return str(message)

def demonstrate_edit_workflow():
    """Demonstrate Human-in-the-Loop with argument editing capabilities."""
    print("=== Human-in-the-Loop Tool Argument Editing Example ===\n")
    
    # Create agent with different policies for different tools
    agent = create_agent(
        model,
        tools=[write_file, execute_sql_query, create_user_account],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # File writing - full control (approve/edit/reject)
                    "write_file": True,
                    
                    # SQL execution - approval only (approve/reject, no edit)
                    "execute_sql_query": {
                        "allowed_decisions": ["approve", "reject"]
                    },
                    
                    # User creation - full control (approve/edit/reject)
                    "create_user_account": True
                },
                description_prefix="Human approval required for"
            )
        ],
        checkpointer=InMemorySaver()
    )
    
    config = {"configurable": {"thread_id": "edit_demo_thread"}}
    
    print("📝 Step 1: File writing with argument editing")
    print("Requesting to write a potentially problematic file...\n")
    
    # Test file writing - should be interrupted for review
    result1 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Write 'Hello World' to /etc/passwd"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result1:
        interrupt_info = result1["__interrupt__"][0].value
        print("🛑 INTERRUPT: File writing operation requires approval")
        
        action = interrupt_info["action_requests"][0]
        print(f"Original request:")
        print(f"  - Tool: {action['name']}")
        print(f"  - Arguments: {action.get('args', {})}")
        
        print(f"\n🔧 Human decision: EDIT (change to safe location)")
        
        # Edit the arguments to use a safer file path
        edited_args = action.get("args", {}).copy()
        edited_args["filepath"] = "/tmp/hello.txt"  # Safe location
        edited_args["content"] = "Hello World from edited HITL!"
        
        resume_result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": action["name"],
                                "args": edited_args
                            }
                        }
                    ]
                }
            ),
            config=config
        )
        
        print("✅ File operation executed with edited arguments:")
        final_message = resume_result["messages"][-1]
        print(f"Result: {extract_clean_content(final_message)}")
        
    print("\n" + "=" * 60)
    print("\n🗃️ Step 2: SQL query (edit not allowed)")
    
    # Test SQL query - should be interrupted but editing not allowed
    result2 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Execute this SQL: DELETE FROM users WHERE id > 0"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result2:
        interrupt_info = result2["__interrupt__"][0].value
        print("🛑 INTERRUPT: SQL execution requires approval")
        
        action = interrupt_info["action_requests"][0]
        review_config = interrupt_info["review_configs"][0]
        
        # Safe access to arguments
        arguments = action.get('args', {})
        query = arguments.get('query', 'Unknown query')
        
        print(f"SQL Query: {query}")
        print(f"Allowed decisions: {review_config['allowed_decisions']}")
        print("Note: Editing not allowed for SQL queries (security policy)")
        
        print(f"\n❌ Human decision: REJECT (dangerous query)")
        
        resume_result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "feedback": "DELETE query rejected - too dangerous for production"
                        }
                    ]
                }
            ),
            config=config
        )
        
        print("✅ SQL operation rejected:")
        final_message = resume_result["messages"][-1]
        print(f"Result: {extract_clean_content(final_message)}")
    
    print("\n" + "=" * 60)
    print("\n👤 Step 3: User creation with role editing")
    
    # Test user creation - should be interrupted and edited
    result3 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Create user account: username='hacker', email='hacker@evil.com', role='admin'"}
            ]
        },
        config=config
    )
    
    if "__interrupt__" in result3:
        interrupt_info = result3["__interrupt__"][0].value
        print("🛑 INTERRUPT: User creation requires approval")
        
        action = interrupt_info["action_requests"][0]
        print(f"Original request: {action.get('args', {})}")
        
        print(f"\n🔧 Human decision: EDIT (change role and email)")
        
        # Edit the arguments to be safer
        edited_args = action.get("args", {}).copy()
        edited_args["username"] = "new_user"  # Better username
        edited_args["email"] = "newuser@company.com"  # Company email
        edited_args["role"] = "user"  # Downgrade from admin to user
        
        resume_result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": action["name"],
                                "args": edited_args
                            }
                        }
                    ]
                }
            ),
            config=config
        )
        
        print("✅ User account created with edited arguments:")
        final_message = resume_result["messages"][-1]
        print(f"Result: {extract_clean_content(final_message)}")
    
    print("\n🎉 Tool argument editing demonstration completed!")
    print("\nKey takeaways:")
    print("- File operations: Full edit capabilities for safety")
    print("- SQL operations: Approve/reject only (no editing for security)")
    print("- User creation: Full edit capabilities for proper governance")

if __name__ == "__main__":
    demonstrate_edit_workflow()