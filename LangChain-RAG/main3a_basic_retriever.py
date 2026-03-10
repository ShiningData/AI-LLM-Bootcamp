"""
Basic Retriever Setup - Part 3a

Shows how to:
- Create a simple retriever from a vector store
- Configure retriever parameters
- Test basic document retrieval
"""
# Chroma requires pysqlite3 instead of sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import tempfile
import shutil
from uuid import uuid4

load_dotenv()

def create_sample_knowledge_base():
    """Create sample documents for retrieval testing."""
    return [
        Document(
            page_content="Python is a versatile programming language used for web development, data analysis, and machine learning.",
            metadata={"topic": "programming", "difficulty": "beginner"}
        ),
        Document(
            page_content="Machine learning algorithms learn patterns from data to make predictions and decisions.",
            metadata={"topic": "ai", "difficulty": "intermediate"}
        ),
        Document(
            page_content="Data preprocessing cleans and prepares raw data for analysis and modeling.",
            metadata={"topic": "data_science", "difficulty": "intermediate"}
        ),
        Document(
            page_content="SQL is used for managing and querying relational databases effectively.",
            metadata={"topic": "database", "difficulty": "beginner"}
        )
    ]

def create_or_get_vectorstore(persist_directory="./vectorstore"):
    """Show basic retriever setup and usage."""
    print("=== Basic Retriever Setup ===\n")

    # Setup embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )
    print("Google Gemini embeddings ready")
    
    vectorstore = Chroma(
        collection_name="retriever-demo",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore

def ingest_documents_to_vectorstore(vectorstore):
    # Add documents
     # Create vector store
    documents = create_sample_knowledge_base()
    doc_ids = [str(uuid4()) for _ in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)
    
    print(f"Added {len(documents)} documents to knowledge base")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.metadata['topic']}: {doc.page_content[:50]}...")
    print()
    
    return True
    
        

def demonstrate_retriever_usage(vectorstore):
    """Show basic retriever usage patterns."""
    print("=== Retriever Usage ===\n")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("Retriever created with k=2 (top 2 results)\n")
    
    # Test queries
    test_queries = [
        "programming languages and coding",
        "machine learning and AI",
        "database management"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Use retriever to get relevant documents
        relevant_docs = retriever.invoke(query)
        
        print(f"Retrieved {len(relevant_docs)} documents:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"  {i}. Topic: {doc.metadata['topic']}")
            print(f"     Content: {doc.page_content[:60]}...")
        print()

if __name__ == "__main__":
    vector_store = create_or_get_vectorstore()
    #ingest_documents_to_vectorstore(vectorstore=vector_store)
    demonstrate_retriever_usage(vectorstore=vector_store)