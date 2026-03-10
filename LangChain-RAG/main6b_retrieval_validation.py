"""
Retrieval Validation for Hybrid RAG - Part 6b

Quality control and validation of retrieved content.
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

class RetrievalValidator:
    def __init__(self, model):
        self.model = model
    
    def validate_retrieval(self, query: str, retrieved_docs: list) -> dict:
        """Validate if retrieved documents are relevant to query."""
        if not retrieved_docs:
            return {"sufficient": False, "reason": "No documents retrieved", "score": 0.0}
        
        # Simple relevance check
        content = "\n".join([doc.page_content for doc in retrieved_docs])
        
        validation_prompt = f"""Rate the relevance of these retrieved documents to the query (0-10):

Query: {query}
Retrieved content: {content[:300]}...

Score (0-10) and brief reason:"""
        
        response = self.model.invoke(validation_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Extract score (simple parsing)
        try:
            score_str = result.split()[0]
            score = float(score_str) / 10.0
        except:
            score = 0.5  # Default middle score
        
        return {
            "sufficient": score > 0.6,
            "score": score,
            "reason": result.strip(),
            "doc_count": len(retrieved_docs)
        }
    
    def check_content_quality(self, docs: list) -> dict:
        """Check quality of retrieved content."""
        if not docs:
            return {"quality": "poor", "issues": ["No content"]}
        
        total_length = sum(len(doc.page_content) for doc in docs)
        avg_length = total_length / len(docs)
        
        issues = []
        if avg_length < 20:
            issues.append("Content too short")
        if len(docs) < 2:
            issues.append("Too few documents")
        if total_length > 1000:
            issues.append("Too much content")
        
        quality = "poor" if issues else "good" if avg_length > 50 else "fair"
        
        return {
            "quality": quality,
            "avg_length": avg_length,
            "doc_count": len(docs),
            "total_length": total_length,
            "issues": issues
        }

def demo_validation():
    print("=== Retrieval Validation Demo ===\n")
    
    # Setup
    docs = [
        Document(page_content="TechCorp Basic Plan costs $29/month with 2 VMs."),
        Document(page_content="Weather is sunny today with mild temperatures."),  # Irrelevant
        Document(page_content="Professional Plan costs $99/month with 10 VMs.")
    ]
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)
    validator = RetrievalValidator(model)
    
    try:
        queries = [
            "What are TechCorp's pricing plans?",
            "What's the weather like?"
        ]
        
        for query in queries:
            print(f"Query: {query}")
            
            # Retrieve documents
            retrieved = vectorstore.similarity_search(query, k=2)
            print(f"Retrieved {len(retrieved)} documents")
            
            # Validate relevance
            validation = validator.validate_retrieval(query, retrieved)
            print(f"Relevance: {'Sufficient' if validation['sufficient'] else 'Insufficient'} (score: {validation['score']:.2f})")
            print(f"Reason: {validation['reason'][:100]}...")
            
            # Check content quality
            quality = validator.check_content_quality(retrieved)
            print(f"Quality: {quality['quality']} ({quality['avg_length']:.0f} avg chars)")
            if quality['issues']:
                print(f"Issues: {', '.join(quality['issues'])}")
            
            print()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def demo_iterative_improvement():
    print("=== Iterative Retrieval Improvement ===\n")
    
    docs = [
        Document(page_content="TechCorp Basic: $29/month"),
        Document(page_content="Professional: $99/month"),  
        Document(page_content="Enterprise: Contact sales"),
        Document(page_content="Support: 24/7 chat, email")
    ]
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)
    validator = RetrievalValidator(model)
    
    try:
        query = "pricing plans"
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            print(f"Attempt {attempt}:")
            
            # Adjust k based on previous results
            k = min(2 + attempt, len(docs))
            retrieved = vectorstore.similarity_search(query, k=k)
            
            validation = validator.validate_retrieval(query, retrieved)
            print(f"Retrieved {len(retrieved)} docs, relevance: {validation['score']:.2f}")
            
            if validation['sufficient']:
                print("Retrieval successful!")
                break
            else:
                print("Insufficient, trying with more documents...")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    demo_validation()
    demo_iterative_improvement()