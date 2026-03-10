"""
Query Enhancement for Hybrid RAG - Part 6a

Intelligent query processing and enhancement.
"""
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tempfile, shutil, numpy as np
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=600)

class MockEmbeddings:
    def embed_documents(self, texts): return [np.random.rand(384).tolist() for _ in texts]
    def embed_query(self, _): return np.random.rand(384).tolist()

class QueryEnhancer:
    def __init__(self, model):
        self.model = model
    
    def enhance_query(self, original_query: str) -> str:
        """Enhance query for better retrieval."""
        enhancement_prompt = f"""Enhance this query for better document retrieval by adding relevant keywords and context:

Original: {original_query}

Enhanced query (add synonyms, expand context, keep concise):"""
        
        response = self.model.invoke(enhancement_prompt)
        enhanced = response.content if hasattr(response, 'content') else str(response)
        return enhanced.strip()
    
    def analyze_query_type(self, query: str) -> dict:
        """Analyze query characteristics."""
        analysis = {
            "complexity": "high" if len(query.split()) > 10 else "medium" if len(query.split()) > 5 else "low",
            "type": "pricing" if any(word in query.lower() for word in ["cost", "price", "$"]) else 
                   "technical" if any(word in query.lower() for word in ["how", "setup", "configure"]) else "general",
            "specificity": "specific" if any(word in query.lower() for word in ["what", "how much", "when"]) else "broad"
        }
        return analysis

def demo_query_enhancement():
    print("=== Query Enhancement Demo ===\n")
    
    enhancer = QueryEnhancer(model)
    
    test_queries = [
        "pricing",
        "VM slow",
        "How do I set up TechCorp for my startup?"
    ]
    
    for query in test_queries:
        print(f"Original: {query}")
        
        # Analyze query
        analysis = enhancer.analyze_query_type(query)
        print(f"Analysis: {analysis}")
        
        # Enhance query
        enhanced = enhancer.enhance_query(query)
        print(f"Enhanced: {enhanced}")
        
        # Show improvement
        improvement = len(enhanced) - len(query)
        print(f"Improvement: +{improvement} characters, more context added\n")

def demo_retrieval_with_enhancement():
    print("=== Enhanced vs Original Retrieval ===\n")
    
    # Setup
    docs = [
        Document(page_content="TechCorp Basic Plan: $29/month, 2 VMs, 100GB storage, email support."),
        Document(page_content="Professional Plan: $99/month, 10 VMs, 500GB storage, priority support."),
        Document(page_content="VM performance issues: Check CPU usage, memory limits, disk space.")
    ]
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)
    enhancer = QueryEnhancer(model)
    
    try:
        test_query = "cost"
        
        # Original retrieval
        print(f"Query: {test_query}")
        original_docs = vectorstore.similarity_search(test_query, k=2)
        print(f"Original retrieval: {len(original_docs)} docs")
        
        # Enhanced retrieval
        enhanced_query = enhancer.enhance_query(test_query)
        print(f"Enhanced query: {enhanced_query}")
        enhanced_docs = vectorstore.similarity_search(enhanced_query, k=2)
        print(f"Enhanced retrieval: {len(enhanced_docs)} docs")
        
        print("\nRetrieved content comparison:")
        print("Original results:", [doc.page_content[:50] + "..." for doc in original_docs])
        print("Enhanced results:", [doc.page_content[:50] + "..." for doc in enhanced_docs])
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    demo_query_enhancement()
    demo_retrieval_with_enhancement()