"""
Basic Agentic RAG - Part 5a

Core agent setup with simple retrieval tool.
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

# Create knowledge base
docs = [
    Document(page_content="TechCorp Basic Plan: $29/month, 2 VMs, 100GB storage."),
    Document(page_content="Professional Plan: $99/month, 10 VMs, 500GB storage."),
    Document(page_content="24/7 support via chat, email, phone.")
]

temp_dir = tempfile.mkdtemp()
vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)

@tool
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for information."""
    docs = vectorstore.similarity_search(query, k=2)
    results = "\n".join([doc.page_content for doc in docs])
    return f"Found: {results}"

@tool  
def calculate_pricing(plan: str, months: int) -> str:
    """Calculate pricing for a plan over time."""
    prices = {"basic": 29, "professional": 99}
    price = prices.get(plan.lower(), 0)
    total = price * months
    return f"{plan} plan for {months} months: ${total}"

def demo_basic_agent():
    print("=== Basic Agentic RAG Demo ===\n")
    
    tools = [search_knowledge_base, calculate_pricing]
    agent = create_agent(model, tools, system_prompt="You are a helpful TechCorp assistant. Use tools to answer questions accurately.")
    
    questions = [
        "What are the pricing plans?",
        "How much would the professional plan cost for 6 months?",
        "What support options are available?"
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
    demo_basic_agent()