"""
Directory Loading and Chunking - Part 1c

Shows how to:
- Load multiple files from directories using DirectoryLoader
- Process different file types in batch
- Handle file filtering and organization
- Manage large document collections efficiently
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader, S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
import shutil

def create_sample_directory():
    """Create a sample directory with multiple files."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Sample documents for different purposes
    documents = {
        "readme.txt": """TechCorp Platform Documentation

                    Welcome to TechCorp's cloud computing platform! This guide will help you get started with our services.

                    Getting Started:
                    1. Create your account at app.techcorp.com
                    2. Choose your subscription plan
                    3. Set up your first project
                    4. Deploy your applications

                    Our platform supports multiple programming languages, databases, and deployment options.""",
        
        "api_guide.txt": """API Authentication Guide

                    Use OAuth2 for secure API access. All requests must include a valid access token in the Authorization header.

                    Authentication Flow:
                    1. Register your application to get client credentials
                    2. Request access token using client credentials grant
                    3. Include token in API requests: Authorization: Bearer <token>
                    4. Refresh tokens before expiration

                    Rate Limits:
                    - Basic Plan: 1000 requests/hour
                    - Professional: 5000 requests/hour
                    - Enterprise: Unlimited""",
        
        "pricing.txt": """Pricing Information

                    Basic Plan: $29/month
                    - 2 virtual machines
                    - 100GB storage
                    - Email support
                    - 99.9% uptime SLA

                    Professional Plan: $99/month
                    - 10 virtual machines
                    - 500GB storage
                    - Priority support
                    - 99.95% uptime SLA

                    Enterprise Plan: Custom pricing
                    - Unlimited virtual machines
                    - Custom storage
                    - Dedicated account manager
                    - 99.99% uptime SLA""",
        
        "faq.txt": """Frequently Asked Questions

                    Q: How do I reset my password?
                    A: Click "Forgot Password" on the login page and follow the email instructions.

                    Q: Can I upgrade my plan anytime?
                    A: Yes, you can upgrade your plan instantly from your dashboard. Billing is prorated.

                    Q: What regions are available?
                    A: We have data centers in US East, US West, Europe, Asia Pacific, and Canada.

                    Q: Do you offer free trials?
                    A: Yes, all new customers get a 30-day free trial with full access to Professional features.""",
        
        "security.txt": """Security Best Practices

                    Account Security:
                    - Enable two-factor authentication
                    - Use strong, unique passwords
                    - Regularly review access logs
                    - Set up login alerts

                    Infrastructure Security:
                    - Keep systems updated
                    - Use encrypted connections (HTTPS/TLS)
                    - Implement proper firewall rules
                    - Monitor for suspicious activity

                    Data Protection:
                    - Encrypt sensitive data at rest
                    - Use secure backup practices
                    - Implement access controls
                    - Regular security audits"""
    }
    
    # Write files to directory
    for filename, content in documents.items():
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
    
    return temp_dir, documents

def demonstrate_basic_directory_loading():
    """Demonstrate loading all files from a directory."""
    print("=== Basic Directory Loading ===\n")
    
    temp_dir, sample_docs = create_sample_directory()
    
    try:
        print(f"Created directory with {len(sample_docs)} files in {temp_dir} directory:")
        for filename in sample_docs.keys():
            print(f"  {filename}")
        print()
        
        # Load all .txt files from directory
        print(f"Loading all .txt files with DirectoryLoader from {temp_dir}")
        loader = DirectoryLoader(
            temp_dir,
            glob="*.txt",
            loader_cls=TextLoader
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents:")
        
        for doc in documents:
            filename = os.path.basename(doc.metadata['source'])
            print(f"  {filename}:")
            print(f"    Length: {len(doc.page_content)} chars")
            print(f"    Preview: {doc.page_content[:60]}...")
        print()
        
        return documents, temp_dir
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def demonstrate_filtered_loading():
    """Show how to load specific file types or patterns."""
    print("=== Filtered File Loading ===\n")
    
    temp_dir, sample_docs = create_sample_directory()
    
    # Add some additional file types for filtering demo
    additional_files = {
        "config.json": '{"api_version": "v1", "timeout": 30}',
        "styles.css": "body { font-family: Arial; }",
        "script.py": "print('Hello World')",
        "notes.md": "# Meeting Notes\n\nDiscussed project timeline."
    }
    
    # Create additional test files in the temp directory
    # Loop through dictionary where keys are filenames and values are file contents
    for filename, content in additional_files.items():
        file_path = os.path.join(temp_dir, filename)  # Build full path to file
        with open(file_path, 'w') as f:  # Open file for writing
            f.write(content)  # Write the content string to the file
    
    try:
        print(f"Directory contains {len(sample_docs) + len(additional_files)} files")
        all_files = list(sample_docs.keys()) + list(additional_files.keys())
        for filename in sorted(all_files):
            print(f"  {filename}")
        print()
        
        # Load only documentation files (.txt and .md)
        print("Loading documentation files (.txt and .md)...")
        doc_loader = DirectoryLoader(
            temp_dir,
            glob="**/*.(txt|md)",
            loader_cls=TextLoader
        )
        
        doc_files = doc_loader.load()
        print(f"Loaded {len(doc_files)} documentation files:")
        for doc in doc_files:
            filename = os.path.basename(doc.metadata['source'])
            extension = os.path.splitext(filename)[1]
            print(f"  {filename} ({extension})")
        print()
        
        # Load only specific files by pattern
        print("Loading FAQ and pricing files...")
        specific_loader = DirectoryLoader(
            temp_dir,
            glob="**/faq.txt",
            loader_cls=TextLoader
        )
        
        pricing_loader = DirectoryLoader(
            temp_dir,
            glob="**/pricing.txt", 
            loader_cls=TextLoader
        )
        
        faq_docs = specific_loader.load()
        pricing_docs = pricing_loader.load()
        
        print(f"FAQ documents: {len(faq_docs)}")
        print(f"Pricing documents: {len(pricing_docs)}")
        print()
        
        return doc_files, temp_dir
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def demonstrate_loading_and_chunking():
    """Show how to load documents and split them into chunks for RAG."""
    print("=== Document Loading and Chunking ===\n")
    
    temp_dir, _ = create_sample_directory()
    
    try:
        # Load all documents
        loader = DirectoryLoader(
            temp_dir,
            glob="*.txt",
            loader_cls=TextLoader
        )
        
        documents = loader.load()
        print(f"Processing {len(documents)} documents in batch...")
        
        # Batch text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        
        all_chunks = splitter.split_documents(documents)
        print(f"Split into {len(all_chunks)} total chunks")
        
        # Organize chunks by source file
        chunks_by_file = {}
        for chunk in all_chunks:
            filename = os.path.basename(chunk.metadata['source'])
            if filename not in chunks_by_file:
                chunks_by_file[filename] = []
            chunks_by_file[filename].append(chunk)
        
        print("\nChunks per file:")
        for filename, chunks in chunks_by_file.items():
            print(f"  {filename}: {len(chunks)} chunks")
        
        # Show sample chunks from different files
        print("\nSample chunks:")
        for filename, chunks in list(chunks_by_file.items())[:3]:
            print(f"  From {filename}:")
            print(f"    {chunks[0].page_content[:60]}...")
        print()
        
        return all_chunks, temp_dir
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def demonstrate_metadata_enrichment():
    """Show how to enrich documents with additional metadata during loading."""
    print("=== Metadata Enrichment ===\n")
    
    temp_dir, sample_docs = create_sample_directory()
    
    try:
        # Load documents with basic metadata
        loader = DirectoryLoader(
            temp_dir,
            glob="*.txt",
            loader_cls=TextLoader
        )
        
        documents = loader.load()
        
        # Enrich with custom metadata based on filename and content
        enriched_docs = []
        for doc in documents:
            filename = os.path.basename(doc.metadata['source'])
            file_stem = os.path.splitext(filename)[0]
            
            # Add metadata based on filename
            enriched_metadata = {
                **doc.metadata,
                'filename': filename,
                'document_type': file_stem,
                'word_count': len(doc.page_content.split()),
                'char_count': len(doc.page_content),
            }
            
            # Add category based on content type
            if 'pricing' in filename.lower() or '$' in doc.page_content:
                enriched_metadata['category'] = 'pricing'
            elif 'api' in filename.lower() or 'authentication' in doc.page_content.lower():
                enriched_metadata['category'] = 'technical'
            elif 'faq' in filename.lower() or 'question' in doc.page_content.lower():
                enriched_metadata['category'] = 'support'
            elif 'security' in filename.lower():
                enriched_metadata['category'] = 'security'
            else:
                enriched_metadata['category'] = 'general'
            
            # Determine document size
            if len(doc.page_content) < 500:
                enriched_metadata['size'] = 'small'
            elif len(doc.page_content) < 1500:
                enriched_metadata['size'] = 'medium'
            else:
                enriched_metadata['size'] = 'large'
            
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=enriched_metadata
            )
            enriched_docs.append(enriched_doc)
        
        print(f"Enriched {len(enriched_docs)} documents with metadata:")
        for doc in enriched_docs:
            print(f"Filename {doc.metadata['filename']}:")
            print(f"    Category: {doc.metadata['category']}")
            print(f"    Size: {doc.metadata['size']} ({doc.metadata['word_count']} words)")
            print("**" * 30)
        print()
        
        return enriched_docs, temp_dir
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def cleanup_directory(temp_dir):
    """Clean up temporary directory."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    
    try:
        # Run demonstrations
        # basic_docs, temp_dir1 = demonstrate_basic_directory_loading()
        # cleanup_directory(temp_dir1)
        
        # filtered_docs, temp_dir2 = demonstrate_filtered_loading()
        # cleanup_directory(temp_dir2)
        
        # load_chunks, temp_dir3 = demonstrate_loading_and_chunking()
        # cleanup_directory(temp_dir3)
        
        enriched_docs, temp_dir4 = demonstrate_metadata_enrichment()
        cleanup_directory(temp_dir4)
        
    except Exception as e:
        print(f"Error: {e}")
        # Clean up any remaining directories
        import glob
        temp_dirs = glob.glob("/tmp/tmp*")
        for temp_dir in temp_dirs:
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass