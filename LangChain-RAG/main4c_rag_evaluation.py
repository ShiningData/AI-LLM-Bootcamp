"""
RAG Quality Evaluation - Part 4c

Simple evaluation metrics and performance testing.
"""
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tempfile, shutil, numpy as np, time
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

class MockEmbeddings:
    def embed_documents(self, texts): return [np.random.rand(384).tolist() for _ in texts]
    def embed_query(self, _): return np.random.rand(384).tolist()

class SimpleRAGEvaluator:
    def __init__(self, documents, embeddings, model):
        self.temp_dir = tempfile.mkdtemp()
        self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=self.temp_dir)
        self.model = model
    
    def evaluate_retrieval(self, query, expected_content, k=3):
        """Test if retrieval finds expected content."""
        docs = self.vectorstore.similarity_search(query, k=k)
        found = any(expected_content.lower() in doc.page_content.lower() for doc in docs)
        return {"query": query, "found_expected": found, "docs_retrieved": len(docs)}
    
    def evaluate_answer_quality(self, query, answer, expected_keywords):
        """Simple keyword-based answer quality check."""
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        score = len(found_keywords) / len(expected_keywords)
        return {"score": score, "found_keywords": found_keywords}
    
    def benchmark_performance(self, query, runs=3):
        """Measure response time."""
        times = []
        for _ in range(runs):
            start = time.time()
            self.vectorstore.similarity_search(query, k=3)
            times.append(time.time() - start)
        return {"avg_time": sum(times) / len(times), "times": times}
    
    def cleanup(self):
        try: shutil.rmtree(self.temp_dir)
        except: pass

def demo_evaluation():
    print("=== RAG Evaluation Demo ===\n")
    
    # Test data
    docs = [
        Document(page_content="TechCorp Basic Plan costs $29/month with 2 VMs and 100GB storage."),
        Document(page_content="Professional Plan costs $99/month with 10 VMs and 500GB storage."),
        Document(page_content="TechCorp offers 24/7 support via chat, email, and phone.")
    ]
    
    evaluator = SimpleRAGEvaluator(docs, MockEmbeddings(), model)
    
    try:
        # Test retrieval accuracy
        print("--- Retrieval Tests ---")
        tests = [
            ("pricing plans", "$29", ["$29", "$99"]),
            ("support options", "24/7", ["chat", "email", "phone"])
        ]
        
        for query, expected, keywords in tests:
            retrieval = evaluator.evaluate_retrieval(query, expected)
            print(f"Query: {query}")
            print(f"Found expected content: {'Yes' if retrieval['found_expected'] else 'No'}")
            print(f"Documents retrieved: {retrieval['docs_retrieved']}")
            print()
        
        # Test performance
        print("--- Performance Test ---")
        perf = evaluator.benchmark_performance("pricing information")
        print(f"Average retrieval time: {perf['avg_time']:.3f}s")
        print(f"Individual times: {[f'{t:.3f}s' for t in perf['times']]}")
        
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    demo_evaluation()