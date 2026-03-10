"""
Context Engineering for RAG - Part 4b

Compare different prompt strategies for better RAG responses.
"""
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import tempfile, shutil, numpy as np
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

class MockEmbeddings:
    def embed_documents(self, texts): return [np.random.rand(384).tolist() for _ in texts]
    def embed_query(self, _): return np.random.rand(384).tolist()

def test_prompt_variants():
    print("=== Prompt Engineering Comparison ===\n")
    
    # Sample context and question
    context = "TechCorp Basic Plan: $29/month, 2 VMs, 100GB storage. Professional Plan: $99/month, 10 VMs, 500GB storage."
    question = "What are the pricing plans?"
    
    # Different prompt strategies
    prompts = {
        "Basic": "Answer: {context}\nQuestion: {question}",
        "Structured": "You are a helpful assistant.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nProvide a clear, specific answer.",
        "Role-based": "You are a TechCorp sales expert.\n\nINFORMATION: {context}\n\nCUSTOMER: {question}\n\nRespond professionally with exact details."
    }
    
    for name, template in prompts.items():
        print(f"--- {name} Prompt ---")
        prompt_text = template.format(context=context, question=question)
        print(f"Length: {len(prompt_text)} chars")
        
        try:
            response = model.invoke(prompt_text)
            answer = response.content if hasattr(response, 'content') else str(response)
            print(f"Answer: {answer[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
        print()

def context_formatting_demo():
    print("=== Context Formatting Strategies ===\n")
    
    docs = [
        "TechCorp Basic: $29/month, 2 VMs",
        "Professional: $99/month, 10 VMs", 
        "All plans include 24/7 support"
    ]
    
    formats = {
        "Simple": "\n".join(docs),
        "Numbered": "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)]),
        "Structured": "\n".join([f"[INFO] {doc}" for doc in docs])
    }
    
    for name, formatted in formats.items():
        print(f"--- {name} Format ---")
        print(f"Context: {formatted}")
        print(f"Length: {len(formatted)} chars\n")

if __name__ == "__main__":
    test_prompt_variants()
    context_formatting_demo()