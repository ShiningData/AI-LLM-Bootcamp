"""
Advanced Retrieval Strategies - Part 3c

Shows how to:
- Implement different retrieval strategies
- Handle retrieval errors and edge cases
- Evaluate retrieval quality and performance
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

def create_diverse_knowledge_base():
    """Create a diverse knowledge base for testing retrieval strategies."""
    return [
        Document(
            page_content="Python programming language supports object-oriented, functional, and procedural paradigms with dynamic typing.",
            metadata={"topic": "programming", "difficulty": "intermediate", "type": "language"}
        ),
        Document(
            page_content="JavaScript enables interactive web development with frameworks like React, Vue, and Angular for frontend applications.",
            metadata={"topic": "programming", "difficulty": "intermediate", "type": "web"}
        ),
        Document(
            page_content="Deep learning neural networks use multiple layers to learn complex patterns from large datasets automatically.",
            metadata={"topic": "ai", "difficulty": "advanced", "type": "deep_learning"}
        ),
        Document(
            page_content="Data preprocessing includes cleaning, normalization, feature selection, and handling missing values before analysis.",
            metadata={"topic": "data_science", "difficulty": "intermediate", "type": "preprocessing"}
        ),
        Document(
            page_content="Docker containers package applications with dependencies for consistent deployment across environments.",
            metadata={"topic": "devops", "difficulty": "intermediate", "type": "containerization"}
        )
    ]

def get_or_create_vectorstore(persist_directory="./vectorstore"):
    """Show different retrieval strategy configurations."""
    print("=== Retrieval Strategies ===\n")

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
    
    return vectorstore
    
def ingest_documents_to_vectorstore(vectorstore): 
      
    # Create knowledge base
    documents = create_diverse_knowledge_base()
    doc_ids = [str(uuid4()) for _ in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Knowledge base ready with {len(documents)} documents")
    return True


def test_different_strategies(vectorstore):
    
    # Test different strategies
    strategies = [
        {"name": "Focused", "k": 2, "description": "Top 2 most relevant"},
        {"name": "Standard", "k": 3, "description": "Balanced approach"},
        {"name": "Comprehensive", "k": 4, "description": "Broader coverage"}
    ]
    
    query = "programming languages and development"
    print(f"Test query: '{query}'\n")
    
    for strategy in strategies:
        print(f"--- {strategy['name']} Strategy ({strategy['description']}) ---")
        retriever = vectorstore.as_retriever(search_kwargs={"k": strategy['k']})
        
        try:
            results = retriever.invoke(query)
            print(f"Retrieved {len(results)} documents:")
            
            for i, doc in enumerate(results, 1):
                topic = doc.metadata.get('topic', 'Unknown')
                print(f"  {i}. {topic}: {doc.page_content[:50]}...")
                
        except Exception as e:
            print(f"Error: {e}")
        print()
    
    return True

def demonstrate_error_handling(vectorstore):
    """Show how to handle retrieval errors gracefully.
        1. Empty query (''): Still returns results (likely default/random matches)
        2. Very long query: Truncates/processes successfully, finds relevant content
        3. SQL injection attempt: Treats it as text, no security issues - finds closest semantic matches
        4. Non-English query (¿Qué es Python?): Cross-language semantic understanding works - correctly finds Python-related content despite Spanish query

       Vector search with embeddings is:
        - Resilient to malformed inputs
        - Secure against injection attacks (treats everything as semantic text)
        - Language-agnostic (embeddings capture meaning across languages)
        - Fault-tolerant (always returns something, doesn't crash)
    """
    print("=== Error Handling ===\n")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Test problematic queries
    error_cases = [
        {"query": "", "issue": "Empty query"},
        {"query": "x" * 1000, "issue": "Very long query"},
        {"query": "SELECT * FROM table", "issue": "SQL-like query"},
        {"query": "¿Qué es Python?", "issue": "Non-English query"}
    ]
    
    for case in error_cases:
        print(f"Testing: {case['issue']}")
        query_display = case['query'][:30] + '...' if len(case['query']) > 30 else case['query']
        print(f"Query: '{query_display}'")
        
        try:
            results = retriever.invoke(case['query'])
            print(f"Success: {len(results)} results retrieved")
            if results:
                print(f"   First result: {results[0].page_content[:40]}...")
        except Exception as e:
            print(f"Handled error: {type(e).__name__}")
        print()

def demonstrate_quality_metrics():
    """Show retrieval quality evaluation concepts."""
    print("=== Quality Metrics ===\n")
    
    print("Key retrieval metrics:")
    metrics = {
        "Precision@k": "Relevant docs / Retrieved docs",
        "Recall@k": "Relevant docs / Total relevant docs",
        "F1-Score": "Harmonic mean of precision and recall",
        "MRR": "Mean Reciprocal Rank of first relevant result"
    }
    
    for metric, description in metrics.items():
        print(f"   • {metric}: {description}")
    print()
    
    # Simulate evaluation
    print("Example evaluation:")
    test_cases = [
        {"query": "Python programming", "relevant": 2, "retrieved": 3, "total_relevant": 3},
        {"query": "Machine learning", "relevant": 1, "retrieved": 3, "total_relevant": 2}
    ]
    
    for case in test_cases:
        precision = case['relevant'] / case['retrieved']
        recall = case['relevant'] / case['total_relevant']
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Query: '{case['query']}'")
        print(f"      Precision@3: {precision:.2f}")
        print(f"      Recall@3: {recall:.2f}")
        print(f"      F1-Score: {f1:.2f}")
    print()
    
    print("Optimization tips:")
    print("   • Monitor retrieval quality over time")
    print("   • Adjust k based on your use case")
    print("   • Test with diverse query types")
    print("   • Use user feedback to improve")

if __name__ == "__main__":
    #vector_store = get_or_create_vectorstore()
    #ingest_documents_to_vectorstore(vectorstore=vector_store)
    #test_different_strategies(vectorstore=vector_store)
    #demonstrate_error_handling(vectorstore=vector_store)
    demonstrate_quality_metrics()
 
