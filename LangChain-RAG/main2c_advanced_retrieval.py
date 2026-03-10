"""
Advanced Retrieval Strategies - Part 2c

Shows how to:
- Use similarity scores for filtering
- Implement retrieval strategies
- Optimize search performance
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

def setup_demo_vectorstore(persist_directory="./vectorstore"):
    """Set up vector store"""
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    # Setup vector store
    vectorstore = Chroma(
        collection_name="advanced-demo",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

def ingest_documents_to_vectorstore(vectorstore):
    """Add documents to existing vectorstore."""
       # Add test documents
    documents = [
        Document(page_content="Machine learning algorithms for data analysis", metadata={"topic": "ai"}),
        Document(page_content="Python programming for web development", metadata={"topic": "programming"}),
        Document(page_content="Data visualization with charts and graphs", metadata={"topic": "data"}),
        Document(page_content="Deep learning neural networks", metadata={"topic": "ai"}),
        Document(page_content="Cooking pasta with tomato sauce", metadata={"topic": "cooking"})
    ]
    
    doc_ids = [str(uuid4()) for _ in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)

    print(f"Added {len(documents)} documents to vector store")
    return True

def demonstrate_similarity_scores(vectorstore):
    """Show similarity search with scores."""
    print("=== Similarity Scores ===\n")

    query = "artificial intelligence and machine learning"
    print(f"Query: '{query}'")
    
    # Get results with scores
    scored_results = vectorstore.similarity_search_with_score(query, k=3)
    
    print("Results with similarity scores (lower = more similar):")
    for i, (doc, score) in enumerate(scored_results, 1):
        similarity_pct = (1 - score) * 100
        print(f"  {i}. Distance: {score:.3f} (Similarity: {similarity_pct:.1f}%)")
        print(f"     Content: {doc.page_content[:50]}...")

def demonstrate_threshold_filtering(vectorstore):
    """Show filtering by similarity threshold.
     In Chroma (and many vector databases), similarity scores work inversely:
        - Lower score = More similar
        - Higher score = Less similar

        So your results actually show:
        1. Deep learning (0.195) - MOST similar to "AI and ML"
        2. Machine learning (0.232) - Second most similar
        3. Data visualization (0.396) - LEAST similar

        This is the correct semantic ranking!

        Why This Happens:

        Chroma uses cosine distance instead of cosine similarity:
        - Cosine similarity: 1.0 = identical, 0.0 = unrelated
        - Cosine distance: 0.0 = identical, 1.0 = unrelated (distance = 1 - similarity)
    """
    print("=== Threshold Filtering ===\n")
    
    queries = [
        ("machine learning models", "Relevant query"),
        ("cooking recipes", "Less relevant query")
    ]
    print()
    
    for query, description in queries:
        print(f"Query: '{query}' ({description})")
        
        scored_results = vectorstore.similarity_search_with_score(query, k=5)
        
        for doc, score in scored_results:
            print(f"Doc: {doc.page_content[:30]}...")
            print(f"Score: {score}")
        print("===========  END OF QUERY RESULT =================")

if __name__ == "__main__":
    vector_store = setup_demo_vectorstore()
    #ingest_documents_to_vectorstore(vectorstore=vector_store)
    demonstrate_similarity_scores(vectorstore=vector_store)
    #demonstrate_threshold_filtering(vectorstore=vector_store)