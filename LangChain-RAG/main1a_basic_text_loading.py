"""
Basic Text Document Loading - Part 1a

Shows how to:
- Load simple text documents using LangChain TextLoader
- Apply different text splitting strategies
- Understand chunking parameters and their effects
- Compare different text splitters
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os

def create_sample_text():
    """Create a sample text document for demonstration."""
    sample_content = """Artificial Intelligence (AI) is a transformative technology that enables machines to perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.

Deep Learning is a specialized area of machine learning that uses neural networks with many layers to model complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Applications include chatbots, translation services, and sentiment analysis.

Computer Vision allows machines to interpret and understand visual information from the world. It powers applications like image recognition, autonomous vehicles, and medical image analysis."""
    
    return sample_content

def demonstrate_basic_text_loading():
    """Demonstrate basic text loading with LangChain TextLoader."""
    print("=== Basic Text Loading ===\n")
    
    # Create temporary text file
    sample_text = create_sample_text()
    
    # Write sample text to /tmp directory
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(sample_text)
        temp_file_path = temp_file.name
    
    try:
        # Load using TextLoader
        print(f"Loading document with TextLoader from {temp_file_path}")
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        
        doc = documents[0]
        print(f"Loaded document:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Preview: {doc.page_content[:100]}...")
        print(f"  Document count: {len(documents)}")
        print()
        
        return documents
        
    finally:
        # Clean up
        os.unlink(temp_file_path)

def demonstrate_recursive_splitting():
    """Demonstrate RecursiveCharacterTextSplitter."""
    print("=== Recursive Text Splitting ===\n")
    
    sample_text = create_sample_text()
    documents = [Document(page_content=sample_text, metadata={"source": "sample.txt"})]
    
    # Small chunks for precise retrieval
    # Splits text using a hierarchy of separators, trying each one in order until chunks are small enough. The default hierarchy is:
    # ["\n\n", "\n", " ", ""]
    # RecursiveCharacterTextSplitter is generally the better default choice because it tries harder to 
    # preserve semantic boundaries (paragraphs → sentences → words) rather than cutting arbitrarily.
    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    small_chunks = small_splitter.split_documents(documents)
    print(f"Small chunks (200 chars, 50 overlap): {len(small_chunks)} chunks")
    # Loop for the first 3 chunks
    for i, chunk in enumerate(small_chunks, 1):
        print(f"  Chunk {i}: {chunk.page_content}")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()
    
    # Medium chunks for balanced context
    medium_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    medium_chunks = medium_splitter.split_documents(documents)
    print(f"Medium chunks (500 chars, 100 overlap): {len(medium_chunks)} chunks")
    # Loop for the first 2 chunks
    for i, chunk in enumerate(medium_chunks[:2], 1):
        print(f"  Chunk {i}: {chunk.page_content}")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()

def demonstrate_character_splitting():
    """Demonstrate CharacterTextSplitter with different separators."""
    print("=== Character Text Splitting ===\n")
    
    sample_text = create_sample_text()
    documents = [Document(page_content=sample_text, metadata={"source": "sample.txt"})]
    
    # Splits text based on a single separator (default is "\n\n"). It tries to split at that separator, 
    # and if chunks are still too large, it just cuts at the character limit.
    paragraph_splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separator="\n\n"
    )
    
    para_chunks = paragraph_splitter.split_documents(documents)
    print(f"Paragraph splitting: {len(para_chunks)} chunks")
    for i, chunk in enumerate(para_chunks, 1):
        print(f"  Chunk {i}: {chunk.page_content[:60]}...")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()
    
    # Split by sentences
    sentence_splitter = CharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30,
        separator=". "
    )
    
    sent_chunks = sentence_splitter.split_documents(documents)
    print(f"Sentence splitting: {len(sent_chunks)} chunks")
    for i, chunk in enumerate(sent_chunks[:3], 1):
        print(f"  Chunk {i}: {chunk.page_content[:60]}...")
        print(f"    Length: {len(chunk.page_content)} chars")
    print()

def demonstrate_chunk_overlap_effects():
    """Show how chunk overlap affects retrieval coverage."""
    print("=== Chunk Overlap Effects ===\n")
    
    sample_text = create_sample_text()
    documents = [Document(page_content=sample_text, metadata={"source": "sample.txt"})]
    
    # No overlap
    no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0
    )
    print("=== Chunks No Overlap ===\n")
    chunks_no_overlap = no_overlap.split_documents(documents)
    print(f"No overlap: {len(chunks_no_overlap)} chunks")
    for i, chunk in enumerate(chunks_no_overlap, 1):
        print(f"Chunk: {i}")
        print(chunk)
    
    # With overlap
    with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50
    )
    
    print("\n\n=== Chunks With Overlap ===\n")
    chunks_with_overlap = with_overlap.split_documents(documents)
    print(f"With 50 char overlap: {len(chunks_with_overlap)} chunks")
    for i, chunk in enumerate(chunks_with_overlap, 1):
        print(f"Chunk: {i}")
        print(chunk)

if __name__ == "__main__":
    print("Basic Text Document Loading - Part 1a")
    print("Focus: Simple text loading and fundamental splitting strategies\n")
    
    #demonstrate_basic_text_loading()
    # demonstrate_recursive_splitting() 
    demonstrate_character_splitting()
    #demonstrate_chunk_overlap_effects()