"""
Adaptive Retrieval - Part 5c

Agent decides when and how to retrieve based on context.
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

# Setup knowledge base
docs = [
    Document(page_content="TechCorp pricing: Basic $29/month, Pro $99/month."),
    Document(page_content="Support: 24/7 chat, email, phone. Enterprise gets dedicated managers."),
    Document(page_content="Security: End-to-end encryption, MFA, SOC2 compliant.")
]

temp_dir = tempfile.mkdtemp()
vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)

@tool
def search_company_info(query: str) -> str:
    """Search for specific company information when needed."""
    docs = vectorstore.similarity_search(query, k=2)
    return f"Company info: {'; '.join([doc.page_content for doc in docs])}"

@tool
def get_current_status() -> str:
    """Check current system status and availability."""
    return "System Status: All services operational. 99.9% uptime this month."

def demo_adaptive():
    print("=== Adaptive Retrieval Demo ===\n")
    
    tools = [search_company_info, get_current_status]
    
    # Agent with adaptive behavior
    system_prompt = """You are a TechCorp assistant. 

Guidelines:
- For pricing/product questions: Use search_company_info
- For status questions: Use get_current_status  
- For general greetings: Answer directly without tools
- Be efficient - only use tools when necessary"""
    
    agent = create_agent(model, tools, system_prompt=system_prompt)
    
    questions = [
        "Hello!",  # Should not use tools
        "What's the current system status?",  # Should use get_current_status
        "How much does the Pro plan cost?",  # Should use search_company_info
        "Thanks for the help!"  # Should not use tools
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
    demo_adaptive()