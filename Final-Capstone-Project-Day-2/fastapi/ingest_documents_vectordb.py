from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_qdrant import Qdrant, QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from fastapi import UploadFile
from typing import List
import os
import tempfile

load_dotenv()


def ingest_documents(directory_path: str, collection_name: str = "my_documents", 
                    difficulty: str = "middle", main_language: str = "Python"):
    """
    Ingest documents from a directory into vector database
    
    Args:
        directory_path: Path to directory containing documents
        collection_name: Name of the Qdrant collection
        difficulty: Difficulty level metadata
        main_language: Main programming language metadata
    
    Returns:
        dict: Result status and message
    """
    try:
        # Load documents from directory
        docs = DirectoryLoader(path=directory_path).load()
        
        if not docs:
            return {"status": "error", "message": f"No documents found in {directory_path}"}

        # Add custom metadata
        for doc in docs:
            doc.metadata['difficulty'] = difficulty
            doc.metadata['main_language'] = main_language
            doc.metadata['ingested_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Split documents into smaller chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # chunk size (characters)
            chunk_overlap=150,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )

        all_splits = text_splitter.split_documents(docs)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Create vector store
        url = "http://vectordb:6333"
        vector_store = QdrantVectorStore.from_documents(
            all_splits,
            embeddings,
            url=url,
            prefer_grpc=True,
            collection_name=collection_name,
        )
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(docs)} documents with {len(all_splits)} chunks into collection '{collection_name}'"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to ingest documents: {str(e)}"}


async def ingest_files(files: List[UploadFile], collection_name: str = "my_documents", 
                      difficulty: str = "middle", main_language: str = "Python"):
    """
    Ingest uploaded files into vector database
    
    Args:
        files: List of uploaded files
        collection_name: Name of the Qdrant collection
        difficulty: Difficulty level metadata
        main_language: Main programming language metadata
    
    Returns:
        dict: Result status and message
    """
    try:
        docs = []
        
        for file in files:
            # Read file content
            content = await file.read()
            text_content = content.decode('utf-8')
            
            # Create document with metadata
            doc = Document(
                page_content=text_content,
                metadata={
                    'source': file.filename,
                    'difficulty': difficulty,
                    'main_language': main_language,
                    'ingested_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            docs.append(doc)
        
        if not docs:
            return {"status": "error", "message": "No files were uploaded"}

        # Split documents into smaller chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(docs)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Create vector store
        url = "http://vectordb:6333"
        vector_store = QdrantVectorStore.from_documents(
            all_splits,
            embeddings,
            url=url,
            prefer_grpc=True,
            collection_name=collection_name,
        )
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(files)} files with {len(all_splits)} chunks into collection '{collection_name}'"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to ingest files: {str(e)}"}
