__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

docs = DirectoryLoader(
   path="./text_files"
).load()

# Add custom metadata
for doc in docs:
    doc.metadata['difficulty']='middle'
    doc.metadata['main_language']='Python'
    doc.metadata['ingested_at']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Split documents into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # chunk size (characters)
    chunk_overlap=150,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

all_splits = text_splitter.split_documents(docs)


embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )


vector_store = Chroma(
    collection_name="vbo-bootcamps",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Ingest chunks into vectore store
document_ids = vector_store.add_documents(documents=all_splits)

# print(len(docs))

# for doc in docs:
#     print(f"Metadata: {doc.metadata}")
#     print(f"Page content: {doc.page_content[:100]}")


# for chunk in all_splits[:5]:
#     print(f"Metadata: {chunk.metadata}")
#     print(f"Page content: {chunk.page_content[:100]}")
#     print("==" * 50)
#     print()