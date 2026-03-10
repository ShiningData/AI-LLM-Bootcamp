"""
Basic Text Embeddings - Part 2a

Shows how to:
- Understand what embeddings are 
- Use Google Gemini embeddings
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def demonstrate_google_gemini():
    """Show Google Gemini embeddings in action."""
    print("=== Google Gemini Embeddings ===\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )
    print("Google Gemini embeddings ready!")
    print(f"Model: text-embedding-004 (768 dimensions)")
    
    # Test with sample texts
    sample_texts = [
        "machine learning algorithms",
        "artificial intelligence research", 
        "cooking recipes and food"
    ]
    
    print("\n Converting texts to embeddings:")
    embeddings_list = []
    
    for text in sample_texts:
        # Convert text string to numerical vector representation using the embedding model
        vector = embeddings.embed_query(text)
        embeddings_list.append(vector)
        print(f"   '{text}'")
        print(f"   → {len(vector)} dimensions")
        print(f"   → Sample: [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, ...]")
        print()
    
    # Show similarity calculation
    import numpy as np
    if len(embeddings_list) >= 2:
        print("Calculating similarities:")
        
        def cosine_similarity(vec1, vec2):
            # Convert lists to numpy arrays for mathematical operations
            v1, v2 = np.array(vec1), np.array(vec2)
            # Calculate cosine similarity: dot product divided by product of vector magnitudes
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Compare first two texts (related) vs first and third (unrelated)
        sim_related = cosine_similarity(embeddings_list[0], embeddings_list[1])
        sim_unrelated = cosine_similarity(embeddings_list[0], embeddings_list[2])
        
        print(f"   'ML algorithms' ↔ 'AI research': {sim_related:.3f} (related)")
        print(f"   'ML algorithms' ↔ 'cooking recipes': {sim_unrelated:.3f} (unrelated)")       
    
    return embeddings

if __name__ == "__main__":
    
    demonstrate_google_gemini()