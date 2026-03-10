"""
Document Loading and Text Splitting - Overview

This is the main overview for document loading examples. 
Each topic has been split into focused, educational modules:

main1a_basic_text_loading.py    - Simple text files and fundamental splitting
main1b_csv_json_loading.py      - Structured data (CSV, JSON) handling
main1c_directory_batch_loading.py - Multiple files and batch processing  
main1d_advanced_splitting.py    - Code-aware and semantic splitting

Run each file individually to learn specific aspects of document loading for RAG.
"""

def print_overview():
    """Print an overview of all document loading examples."""
    print("Document Loading and Text Splitting - Complete Guide")
    print("=" * 60)
    print()
    
    examples = [
        {
            "file": "main1a_basic_text_loading.py",
            "title": "Basic Text Loading",
            "description": "Learn fundamental text loading and splitting strategies",
            "topics": [
                "TextLoader for simple text files",
                "RecursiveCharacterTextSplitter vs CharacterTextSplitter", 
                "Chunk size and overlap effects",
                "Understanding splitting parameters"
            ]
        },
        {
            "file": "main1b_csv_json_loading.py", 
            "title": "Structured Data Loading",
            "description": "Handle CSV, JSON, and other structured formats",
            "topics": [
                "CSVLoader with custom source columns",
                "JSONLoader with jq schemas and fallbacks",
                "Processing structured data for RAG",
                "Metadata enrichment and filtering"
            ]
        },
        {
            "file": "main1c_directory_batch_loading.py",
            "title": "Directory and Batch Loading", 
            "description": "Efficiently process multiple files and large collections",
            "topics": [
                "DirectoryLoader with file filtering",
                "Batch processing multiple documents",
                "File type filtering with glob patterns", 
                "Metadata-based organization"
            ]
        },
        {
            "file": "main1d_advanced_splitting.py",
            "title": "Advanced Text Splitting",
            "description": "Handle complex documents with code and mixed content",
            "topics": [
                "Code-aware splitting strategies",
                "Mixed format document handling",
                "Semantic boundary preservation",
                "Content-type optimized splitting"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   File: {example['file']}")
        print(f"   {example['description']}")
        print(f"   Topics covered:")
        for topic in example['topics']:
            print(f"   • {topic}")
        print()
    
    print("Getting Started:")
    print("   Run each example individually to focus on specific concepts:")
    print(f"   python main1a_basic_text_loading.py")
    print(f"   python main1b_csv_json_loading.py") 
    print(f"   python main1c_directory_batch_loading.py")
    print(f"   python main1d_advanced_splitting.py")
    print()
    
    print("Quick Reference:")
    print("   • TextLoader: Simple text files")
    print("   • CSVLoader: Tabular data with metadata")
    print("   • JSONLoader: Structured JSON documents")
    print("   • DirectoryLoader: Multiple files from directories")
    print("   • RecursiveCharacterTextSplitter: Recommended for most cases")
    print("   • CharacterTextSplitter: When you need specific separators")
    print()
    
    print("Best Practices:")
    print("   1. Start with main1a for basic concepts")
    print("   2. Choose appropriate chunk size (200-800 chars typically)")
    print("   3. Use overlap (20-100 chars) to prevent information loss")
    print("   4. Enrich metadata during loading for better retrieval")
    print("   5. Test splitting strategies with your specific content")
    print()

def demonstrate_quick_example():
    """Show a minimal quick-start example."""
    print("Quick Start Example")
    print("-" * 30)
    print()
    
    print("Here's the simplest way to load and split a document:")
    print()
    
    code_example = '''from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a text file
loader = TextLoader("your_document.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks")'''
    
    print(code_example)
    print()
    print("This basic pattern works for most RAG applications!")
    print("   Explore the detailed examples to learn advanced techniques.")
    print()

if __name__ == "__main__":
    print_overview()
    demonstrate_quick_example()
    
    print("Next Steps:")
    print("   • Run the individual examples to dive deeper")
    print("   • Experiment with different chunk sizes for your content") 
    print("   • Try the examples with your own documents")
    print("   • Move on to main2_embeddings_vectorstore.py for the next step")