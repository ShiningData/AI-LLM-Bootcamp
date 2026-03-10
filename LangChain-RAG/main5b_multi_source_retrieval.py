"""
Multi-Source Retrieval - Part 5b

Multiple retrieval tools for different data sources.
"""
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tempfile, shutil, numpy as np
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=600)

class MockEmbeddings:
    def embed_documents(self, texts): return [np.random.rand(384).tolist() for _ in texts]
    def embed_query(self, _): return np.random.rand(384).tolist()

# Create multiple knowledge bases
temp_dir = tempfile.mkdtemp()

# Technical docs
tech_docs = [Document(page_content="VM troubleshooting: Check CPU limits, memory usage, disk space.")]
tech_store = Chroma.from_documents(tech_docs, MockEmbeddings(), persist_directory=f"{temp_dir}/tech")

# Product info  
product_docs = [Document(page_content="TechCorp offers cloud VMs, databases, AI/ML tools.")]
product_store = Chroma.from_documents(product_docs, MockEmbeddings(), persist_directory=f"{temp_dir}/product")

@tool
def search_technical_docs(query: str) -> str:
    """Search technical documentation and troubleshooting guides."""
    docs = tech_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

@tool
def search_product_info(query: str) -> str:
    """Search product information and features."""
    docs = product_store.similarity_search(query, k=2)  
    return "\n".join([doc.page_content for doc in docs])

@tool
def search_web_info(query: str) -> str:
    """Search general web information (simulated)."""
    return f"Web search for '{query}': General cloud computing best practices available online."

def demo_multi_source():
    print("=== Multi-Source Retrieval Demo ===\n")
    
    tools = [search_technical_docs, search_product_info, search_web_info]
    agent = create_agent(model, tools, system_prompt="You are a TechCorp expert. Choose the right tools based on the question type.")
    
    questions = [
        "My VM is running slow, what should I check?",
        "What products does TechCorp offer?",
        "What are cloud computing best practices?"
    ]
    
    try:
        for question in questions:
            print(f"Q: {question}")
            response = agent.invoke({"messages": [{"role": "user", "content": question}]})
            if hasattr(response, 'content'):
                print(f"A: {response.content}\n")
            elif isinstance(response, dict) and 'messages' in response:
                print(f"A: {response['messages'][-1].content}\n")
            else:
                print(f"A: {response}\n")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    demo_multi_source()