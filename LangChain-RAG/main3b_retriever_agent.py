"""
Retriever with Agent Integration - Part 3b

Shows how to:
- Create retrieval tools for agents
- Integrate retrievers with LLM agents
- Handle retrieval in conversational AI
"""
# Chroma requires pysqlite3 instead of sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import tempfile
import shutil
from uuid import uuid4

load_dotenv()

def extract_clean_content(message_content):
    """Extract clean text from agent response."""
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
        return ' '.join(text_parts) if text_parts else str(message_content)
    return str(message_content)

def create_tech_knowledge_base():
    """Create a focused knowledge base for agent testing."""
    return [
        Document(
            page_content="Python is ideal for beginners due to its readable syntax and extensive community support. Popular for data science, web development, and automation.",
            metadata={"topic": "programming", "language": "python"}
        ),
        Document(
            page_content="Machine learning models like regression, decision trees, and neural networks help computers learn from data to make predictions.",
            metadata={"topic": "ai", "category": "machine_learning"}
        ),
        Document(
            page_content="Git version control tracks code changes, enables collaboration through branches, and maintains project history with commits.",
            metadata={"topic": "programming", "category": "version_control"}
        )
    ]

def setup_retrieval_agent(persist_directory="./vectorstore"):
    """Set up an agent with retrieval capabilities."""
    print("=== Retrieval Agent Setup ===\n")
    

    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )
    print("Google Gemini embeddings ready")
    
    vectorstore = Chroma(
        collection_name="agent-kb",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("Retriever configured for agent use\n")
    
    return vectorstore, retriever

def ingest_documents_to_vectorstore(vectorstore): 
      
    # Create knowledge base
    documents = create_tech_knowledge_base()
    doc_ids = [str(uuid4()) for _ in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Knowledge base ready with {len(documents)} documents")
    return True
    

def demonstrate_retrieval_tool(retriever):
    """Show how to create a retrieval tool for agents."""
    print("=== Creating Retrieval Tool ===\n")
    
    # Define retrieval tool
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the tech knowledge base for programming, AI, and development information."""
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found."
            
            results = []
            for i, doc in enumerate(docs, 1):
                result = f"{i}. {doc.page_content}\n   (Topic: {doc.metadata.get('topic', 'N/A')})"
                results.append(result)
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Search error: {e}"
    
    print("Retrieval tool created")
    print("   Function: search_knowledge_base(query)")
    print("   Purpose: Search tech knowledge for relevant info\n")
    
    return search_knowledge_base

def demonstrate_agent_integration(vector_search_tool):
    """Show agent using retrieval for answering questions."""
    print("=== Agent Integration ===\n")
    

    # Initialize model
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        max_output_tokens=400,
        temperature=0
    )
    
    # Create agent with retrieval tool
    agent = create_agent(
        model,
        tools=[vector_search_tool],
        system_prompt="""You are a helpful tech assistant. Use the search tool to find relevant information before answering questions.
        
When answering:
1. Always search the knowledge base first
2. Base your response on the retrieved information
3. Be concise and helpful
4. Mention the source topics when relevant"""
    )
    
    print("Agent created with retrieval capability")
    
    # Test questions
    test_questions = [
        "What can you tell me about Python?",
        "How does machine learning work?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            
            response = extract_clean_content(result['messages'][-1].content)
            print(f"Answer: {response[:200]}...")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    vector_store, retriever = setup_retrieval_agent()
    ingest_documents_to_vectorstore(vectorstore=vector_store)
    vector_search_tool = demonstrate_retrieval_tool(retriever=retriever)
    
    # Test the search tool directly
    print("\n=== Testing Search Tool Directly ===")
    print(f"Python search: {vector_search_tool.invoke('What can you tell me about Python?')}")
    print(f"ML search: {vector_search_tool.invoke('How does machine learning work?')}")
    
    demonstrate_agent_integration(vector_search_tool=vector_search_tool)

