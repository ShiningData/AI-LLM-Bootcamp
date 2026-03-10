"""
RAG Agent Module
================
Creates a RAG agent with short-term memory for conversational HR Q&A.

Key Features:
- Uses LangChain v1.2.0 create_agent API
- Short-term memory for conversation context
- Retrieval tool for searching HR documents
"""
from typing import List, Tuple

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

import vector_store


# ============================================================================
# SHORT-TERM MEMORY
# ============================================================================

class ConversationMemory:
    """
    Simple short-term memory for storing conversation history.
    
    This allows the agent to:
    - Remember previous questions and answers
    - Understand references like "it", "them", "that policy"
    - Maintain context across multiple turns
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize memory.
        
        Args:
            max_messages: Maximum number of message pairs to keep
        """
        self.messages: List = []
        self.max_messages = max_messages
    
    def add_user_message(self, content: str):
        """Add a user message to memory."""
        self.messages.append(HumanMessage(content=content))
        self._trim_if_needed()
    
    def add_ai_message(self, content: str):
        """Add an AI response to memory."""
        self.messages.append(AIMessage(content=content))
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Keep only the last max_messages pairs."""
        max_items = self.max_messages * 2  # pairs of user + ai
        if len(self.messages) > max_items:
            self.messages = self.messages[-max_items:]
    
    def get_messages(self) -> List:
        """Get all messages in memory."""
        return self.messages.copy()
    
    def clear(self):
        """Clear all messages from memory."""
        self.messages = []


# ============================================================================
# RETRIEVAL TOOL
# ============================================================================

@tool(response_format="content_and_artifact")
def search_hr_documents(query: str) -> Tuple[str, List[dict]]:
    """
    Search HR documents for relevant information.
    
    Use this tool to find information about:
    - Leave policy (vacation, sick leave, parental leave)
    - Employee handbook and guidelines
    - IT security requirements
    - Onboarding and offboarding procedures
    - Performance reviews
    - Travel and expense policies
    - Recruitment procedures
    
    Args:
        query: What to search for in HR documents
        
    Returns:
        Relevant document excerpts with sources
    """
    # Search vector store
    results = vector_store.search(query, k=4)
    
    if not results:
        return "No relevant documents found.", []
    
    # Format results for the agent
    formatted_parts = []
    artifacts = []
    
    for doc in results:
        source = doc.metadata.get("file_name", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        
        formatted_parts.append(
            f"[Source: {source}, Chunk: {chunk_idx}]\n{doc.page_content}"
        )
        
        artifacts.append({
            "source": source,
            "chunk_index": chunk_idx,
            "content": doc.page_content[:200] + "..."
        })
    
    return "\n\n---\n\n".join(formatted_parts), artifacts


# ============================================================================
# AGENT CREATION
# ============================================================================

# System prompt - concise to get shorter answers
SYSTEM_PROMPT = """You are an HR Assistant with short-term memory.

Rules:
1. For HR policy questions: Use search_hr_documents tool, cite sources
2. For follow-up questions: Use conversation context (e.g., "them", "it", "that policy")
3. For personal/conversational questions: Answer directly from memory, NO tool needed
4. Keep answers SHORT (2-3 sentences max)

Examples:
- "What is leave policy?" → Search documents, cite source
- "What about sick leave?" → Use context from previous answer + search if needed  
- "My name is Ali" then "What is my name?" → Answer "Ali" from memory, no search
- "Can I carry them over?" → Understand "them" from context, then search
"""


def create_hr_agent(model_name: str = "openai:gpt-4o-mini"):
    """
    Create the HR RAG agent.
    
    Args:
        model_name: Model to use (e.g., "openai:gpt-4o-mini", "google_genai:gemini-2.0-flash")
        
    Returns:
        Compiled agent
    """
    # Initialize the language model
    model = init_chat_model(model_name)
    
    # Create agent with tools
    agent = create_agent(
        model=model,
        tools=[search_hr_documents],
        system_prompt=SYSTEM_PROMPT
    )
    
    return agent


def chat(agent, question: str, memory: ConversationMemory) -> str:
    """
    Send a question to the agent with conversation memory.
    
    Args:
        agent: The HR agent
        question: User's question
        memory: Conversation memory instance
        
    Returns:
        Agent's response
    """
    # Build messages with history
    messages = memory.get_messages() + [HumanMessage(content=question)]
    
    # Get response
    result = agent.invoke({"messages": messages})
    
    # Extract answer
    answer = result["messages"][-1].content
    
    # Save to memory
    memory.add_user_message(question)
    memory.add_ai_message(answer)
    
    return answer