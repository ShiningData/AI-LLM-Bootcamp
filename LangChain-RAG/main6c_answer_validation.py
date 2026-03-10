"""
Answer Validation for Hybrid RAG - Part 6c

Quality control for generated answers and iterative improvement.
"""
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import tempfile, shutil, numpy as np
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=600)

class MockEmbeddings:
    def embed_documents(self, texts): return [np.random.rand(384).tolist() for _ in texts]
    def embed_query(self, _): return np.random.rand(384).tolist()

class AnswerValidator:
    def __init__(self, model):
        self.model = model
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context: {context}"),
            ("human", "{question}")
        ])
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer from query and context."""
        formatted = self.rag_prompt.format_messages(context=context, question=query)
        response = self.model.invoke(formatted)
        return response.content if hasattr(response, 'content') else str(response)
    
    def validate_answer(self, query: str, answer: str, context: str) -> dict:
        """Validate answer quality and relevance."""
        validation_prompt = f"""Evaluate this RAG answer on a scale of 1-10:

Query: {query}
Context: {context[:200]}...
Answer: {answer}

Rate (1-10) for:
- Relevance to query
- Grounding in context  
- Completeness
- Clarity

Overall score and brief feedback:"""
        
        response = self.model.invoke(validation_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Simple score extraction
        try:
            lines = result.strip().split('\n')
            score_line = [line for line in lines if any(char.isdigit() for char in line)][0]
            score = float(''.join(filter(str.isdigit, score_line))) / 10.0
        except:
            score = 0.5
        
        return {
            "score": score,
            "acceptable": score > 0.6,
            "feedback": result.strip(),
            "answer_length": len(answer)
        }
    
    def improve_answer(self, query: str, context: str, previous_answer: str, feedback: str) -> str:
        """Improve answer based on feedback."""
        improvement_prompt = f"""Improve this RAG answer based on feedback:

Query: {query}
Context: {context}
Previous Answer: {previous_answer}
Feedback: {feedback}

Improved Answer:"""
        
        response = self.model.invoke(improvement_prompt)
        return response.content if hasattr(response, 'content') else str(response)

def demo_answer_validation():
    print("=== Answer Validation Demo ===\n")
    
    # Setup
    docs = [
        Document(page_content="TechCorp Basic Plan: $29/month, includes 2 VMs, 100GB storage, email support."),
        Document(page_content="Professional Plan: $99/month, includes 10 VMs, 500GB storage, priority support.")
    ]
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)
    validator = AnswerValidator(model)
    
    try:
        query = "What are the differences between TechCorp plans?"
        
        # Get context
        retrieved = vectorstore.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in retrieved])
        
        # Generate initial answer
        answer = validator.generate_answer(query, context)
        print(f"Query: {query}")
        print(f"Initial Answer: {answer}\n")
        
        # Validate answer
        validation = validator.validate_answer(query, answer, context)
        print(f"Validation Score: {validation['score']:.2f}")
        print(f"Acceptable: {'Yes' if validation['acceptable'] else 'No'}")
        print(f"Feedback: {validation['feedback'][:150]}...\n")
        
        # Improve if needed
        if not validation['acceptable']:
            improved = validator.improve_answer(query, context, answer, validation['feedback'])
            print(f"Improved Answer: {improved}")
            
            # Re-validate
            new_validation = validator.validate_answer(query, improved, context)
            print(f"New Score: {new_validation['score']:.2f}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def demo_iterative_refinement():
    print("=== Iterative Answer Refinement ===\n")
    
    docs = [Document(page_content="TechCorp offers cloud computing services.")]
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(docs, MockEmbeddings(), persist_directory=temp_dir)
    validator = AnswerValidator(model)
    
    try:
        query = "What specific services does TechCorp offer?"
        max_iterations = 2
        
        # Get context (limited)
        retrieved = vectorstore.similarity_search(query, k=1)
        context = "\n".join([doc.page_content for doc in retrieved])
        print(f"Context: {context}")
        print(f"Query: {query}\n")
        
        current_answer = validator.generate_answer(query, context)
        
        for iteration in range(1, max_iterations + 1):
            print(f"Iteration {iteration}:")
            print(f"Answer: {current_answer}")
            
            validation = validator.validate_answer(query, current_answer, context)
            print(f"Score: {validation['score']:.2f}")
            
            if validation['acceptable']:
                print("Answer meets quality threshold!")
                break
            else:
                print("Needs improvement...")
                current_answer = validator.improve_answer(query, context, current_answer, validation['feedback'])
            
            print()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    demo_answer_validation()
    demo_iterative_refinement()