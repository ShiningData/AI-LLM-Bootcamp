"""
Vector Store Module
===================
Manages ChromaDB vector database for storing and retrieving document embeddings.
"""
# Fix for SQLite version issue on some systems
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from typing import List, Optional, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Configuration
COLLECTION_NAME = "vbo-aillm-bc-rag"
PERSIST_DIRECTORY = "./chroma_db"


def get_embeddings():
    """
    Get the embedding model for converting text to vectors.
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )


def get_vector_store() -> Chroma:
    """
    Get or create the ChromaDB vector store.
    
    Returns:
        Chroma vector store instance
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIRECTORY
    )


def add_documents(documents: List[Document]) -> List[str]:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of documents to add
        
    Returns:
        List of document IDs
    """
    store = get_vector_store()
    ids = store.add_documents(documents)
    print(f"✅ Added {len(ids)} documents to vector store")
    return ids


def search(query: str, k: int = 4) -> List[Document]:
    """
    Search for similar documents.
    
    Args:
        query: Search query text
        k: Number of results to return
        
    Returns:
        List of similar documents
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k)
    return results


def delete_by_filename(filename: str) -> bool:
    """
    Delete all chunks for a specific file.
    
    Used when updating documents - delete old version before adding new.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        True if deletion was successful
    """
    store = get_vector_store()
    collection = store._collection
    
    # Find documents with this filename
    results = collection.get(where={"file_name": filename})
    
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"✅ Deleted {len(results['ids'])} chunks for {filename}")
        return True
    
    return False


def get_document_hash(filename: str) -> Optional[str]:
    """
    Get the stored hash for a document.
    
    Used to check if a document has changed.
    
    Args:
        filename: Name of the file
        
    Returns:
        Document hash or None if not found
    """
    store = get_vector_store()
    collection = store._collection
    
    results = collection.get(
        where={"file_name": filename},
        include=["metadatas"]
    )
    
    if results and results["metadatas"]:
        return results["metadatas"][0].get("document_hash")
    
    return None


def get_stats() -> Dict:
    """
    Get statistics about the vector store.
    
    Returns:
        Dictionary with collection statistics
    """
    store = get_vector_store()
    collection = store._collection
    
    count = collection.count()
    results = collection.get(include=["metadatas"])
    
    filenames = set()
    if results and results["metadatas"]:
        for metadata in results["metadatas"]:
            if "file_name" in metadata:
                filenames.add(metadata["file_name"])
    
    return {
        "total_chunks": count,
        "unique_files": len(filenames),
        "files": sorted(list(filenames))
    }