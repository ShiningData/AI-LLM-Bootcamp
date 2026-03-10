"""
Chroma Vector Store Operations - Part 2b

Shows how to:
- Create Chroma vector store
- Add documents with embeddings
- Search for similar documents
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

def create_test_documents():
    """Create simple test documents."""
    return [
        Document(
            page_content="Python is a programming language for data science.",
            metadata={"topic": "programming", "difficulty": "beginner"}
        ),
        Document(
            page_content="Machine learning helps computers learn from data.",
            metadata={"topic": "ai", "difficulty": "intermediate"}
        ),
        Document(
            page_content="Data visualization creates charts from datasets.",
            metadata={"topic": "data", "difficulty": "beginner"}
        )
    ]

def embeddings_and_chroma_setup(persist_directory="./vectorstore"):
    """Create persistent Chroma vectorstore."""
    print("=== Persistent Chroma Vector Store ===\n")
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )
    print("Google Gemini embeddings ready")
        
    # Always use persistent directory
    print(f"Using persistent directory: {persist_directory}")
        
    vectorstore = Chroma(
        collection_name="test-docs",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Check if vectorstore has existing data
    try:
        doc_count = len(vectorstore.get()['ids'])
        if doc_count > 0:
            print(f"Loaded existing vectorstore with {doc_count} documents")
        else:
            print("Vectorstore is empty (ready for ingestion)")
    except:
        print("New vectorstore created (ready for ingestion)")
    
    return vectorstore

def ingest_documents_to_vectorstore(vectorstore):
    """Add documents to existing vectorstore."""
    print("\n=== Adding Documents to Existing Vector Store ===\n")
    
    try:
        documents = create_test_documents()
        print(f"Adding {len(documents)} documents:")
        
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc.metadata['topic']}: {doc.page_content[:40]}...")
        
        # Add documents with IDs
        doc_ids = [str(uuid4()) for _ in documents]
        vectorstore.add_documents(documents=documents, ids=doc_ids)
        
        print(f"Added {len(documents)} documents to vector store")
        return True
        
    except Exception as e:
        print(f"Error adding documents: {e}")
        return False


def demonstrate_similarity_search(vectorstore, query="programming languages and coding", k=2):
    """Show similarity search on existing vector store."""
    print(f"\n=== Similarity Search ===\n")
    print(f"Query: '{query}'")
    
    try:
        # Perform search on existing vectorstore (no re-ingestion)
        results = vectorstore.similarity_search(query, k=k)
        
        print(f"Found {len(results)} similar documents:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata['topic']}: {doc.page_content[:50]}...")
        print()
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def demonstrate_reusable_search(vectorstore):
    """Show how to reuse vectorstore for multiple searches."""
    print("\n" + "="*50)
    print("Performing searches on persistent vectorstore:")
    print("="*50)
    
    # Perform multiple searches on the same vectorstore
    test_queries = [
        "programming languages and coding",
        "artificial intelligence learning", 
        "data analysis and charts"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Search #{i}]")
        demonstrate_similarity_search(vectorstore, query)

def cleanup_vectorstore():
    """Clear the global vectorstore reference (vectorstore files remain on disk)."""
    global _global_vectorstore
    _global_vectorstore = None
    print("Reset global vectorstore reference")


if __name__ == "__main__":
    vector_store = embeddings_and_chroma_setup()
    ingest_documents_to_vectorstore(vectorstore=vector_store) # Comment for second run
    demonstrate_reusable_search(vectorstore=vector_store)