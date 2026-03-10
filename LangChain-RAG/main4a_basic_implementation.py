"""
Basic Two-Step RAG Implementation - Part 4a

Core TwoStepRAGSystem class and simple demonstration.
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
    def embed_query(self, text): return np.random.rand(384).tolist()

class TwoStepRAGSystem:
    def __init__(self, documents, embeddings, model):
        self.temp_dir = tempfile.mkdtemp()
        self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=self.temp_dir)
        self.model = model
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using context: {context}"), ("human", "{question}")
        ])
    
    def answer_question(self, query: str, k: int = 3):
        # Step 1: Retrieve
        docs = self.vectorstore.similarity_search(query, k=k)
        print(f"Retrieved {len(docs)} documents")
        
        # Step 2: Generate
        context = "\n".join([doc.page_content for doc in docs])
        response = self.model.invoke(self.rag_prompt.format_messages(context=context, question=query))
        return response.content if hasattr(response, 'content') else str(response)
    
    def cleanup(self):
        try: shutil.rmtree(self.temp_dir)
        except: pass

def demo():
    docs = [Document(page_content="TechCorp Basic Plan: $29/month, 2 VMs, 100GB storage.")]
    embeddings = MockEmbeddings()
    rag = TwoStepRAGSystem(docs, embeddings, model)
    
    try:
        print("=== Basic Two-Step RAG Demo ===")
        answer = rag.answer_question("What does the Basic plan cost?")
        print(f"Answer: {answer}")
    finally:
        rag.cleanup()

if __name__ == "__main__":
    demo()