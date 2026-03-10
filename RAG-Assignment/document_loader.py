"""
Document Loader Module
======================
Loads HR documents from a directory using LangChain's DirectoryLoader.
Supports DOCX, PDF, and TXT files.
"""
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of a file for change detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash string
    """
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_documents(directory: str) -> List[Document]:
    """
    Load all documents from a directory.
    
    Uses DirectoryLoader to automatically detect and load
    DOCX, PDF, and TXT files.
    
    Args:
        directory: Path to the documents directory
        
    Returns:
        List of loaded documents
    """
    loader = DirectoryLoader(
        path=directory,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from {directory}")
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    
    return chunks


def get_file_stats(file_path: Path) -> dict:
    """
    Get comprehensive file statistics.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file statistics
    """
    stats = file_path.stat()
    
    # Read file content for character count
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1", errors="ignore")
            char_count = len(text_content)
    except Exception:
        char_count = 0
    
    return {
        "file_size_bytes": stats.st_size,
        "character_count": char_count,
        "creation_date": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
    }


def get_document_type(extension: str) -> str:
    """Map file extension to document type category."""
    mapping = {
        ".docx": "document",
        ".doc": "document",
        ".pdf": "pdf",
        ".txt": "text"
    }
    return mapping.get(extension.lower(), "unknown")


def add_metadata(chunks: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Add comprehensive metadata to each chunk (14 fields as per assignment).
    
    Metadata fields:
    - file_name: Original filename for update tracking
    - file_extension: File extension (e.g., .docx, .pdf, .txt)
    - file_size_bytes: Original file size in bytes
    - character_count: Total character count of the document
    - chunk_index: Position within the document
    - chunk_size: Size of the current chunk in characters
    - chunk_overlap: Overlap size used during chunking
    - document_type: File format category (document, text, pdf)
    - creation_date: Original file creation timestamp
    - last_modified: Original file last modified timestamp
    - ingestion_timestamp: When the document was ingested
    - document_hash: SHA256 hash for change detection
    - page_number: Page number (for PDF files)
    - section_title: Section or heading title if available
    
    Args:
        chunks: List of document chunks
        chunk_size: Chunk size used during splitting
        chunk_overlap: Overlap used during splitting
        
    Returns:
        Chunks with comprehensive metadata
    """
    # Group chunks by source file
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)
    
    # Add metadata to each chunk
    enhanced_chunks = []
    ingestion_time = datetime.now().isoformat()
    
    for source, source_chunks in chunks_by_source.items():
        file_path = Path(source)
        
        # Get file statistics
        if file_path.exists():
            file_stats = get_file_stats(file_path)
            file_hash = calculate_file_hash(source)
        else:
            file_stats = {
                "file_size_bytes": 0,
                "character_count": 0,
                "creation_date": "",
                "last_modified": ""
            }
            file_hash = ""
        
        for idx, chunk in enumerate(source_chunks):
            # Extract section title from content (first line if it looks like a header)
            content_lines = chunk.page_content.strip().split("\n")
            first_line = content_lines[0] if content_lines else ""
            section_title = first_line[:50] if len(first_line) < 100 else ""
            
            # Get page number from existing metadata (PDF loaders add this)
            page_number = chunk.metadata.get("page", chunk.metadata.get("page_number", 0))
            
            chunk.metadata.update({
                # File identification
                "file_name": file_path.name,
                "file_extension": file_path.suffix,
                "document_type": get_document_type(file_path.suffix),
                
                # File statistics
                "file_size_bytes": file_stats["file_size_bytes"],
                "character_count": file_stats["character_count"],
                
                # Chunk information
                "chunk_index": idx,
                "chunk_size": len(chunk.page_content),
                "chunk_overlap": chunk_overlap,
                
                # Timestamps
                "creation_date": file_stats["creation_date"],
                "last_modified": file_stats["last_modified"],
                "ingestion_timestamp": ingestion_time,
                
                # Change detection
                "document_hash": file_hash,
                
                # Content context
                "page_number": page_number,
                "section_title": section_title
            })
            enhanced_chunks.append(chunk)
    
    print(f"✅ Added 14 metadata fields to {len(enhanced_chunks)} chunks")
    return enhanced_chunks


def process_documents(directory: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Complete pipeline: load, split, and add metadata.
    
    Args:
        directory: Path to the documents directory
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Processed document chunks ready for embedding
    """
    # Step 1: Load documents
    documents = load_documents(directory)
    
    # Step 2: Split into chunks
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    
    # Step 3: Add comprehensive metadata (14 fields)
    chunks_with_metadata = add_metadata(chunks, chunk_size, chunk_overlap)
    
    return chunks_with_metadata