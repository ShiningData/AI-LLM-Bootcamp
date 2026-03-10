# Retrieval

Large language models (LLMs) are powerful, but they have two key limitations:
- **Finite context** — they can't ingest entire corpora at once
- **Static knowledge** — their training data is frozen at a point in time

Retrieval addresses these problems by fetching relevant external knowledge at query time. This is the foundation of **Retrieval-Augmented Generation (RAG)**: enhancing an LLM's answers with context-specific information.

## 🛠️ Prerequisites

```bash
# Install required packages
pip install langchain python-dotenv faiss-cpu numpy

# Set Google API key
export GOOGLE_API_KEY="your-api-key-here"
```

## 📁 Examples

### 1️⃣ [main1_document_loading.py](./main1_document_loading.py)
**Document Loading and Text Splitting - Overview**

Comprehensive guide to document loading with focused sub-modules:
- **[main1a_basic_text_loading.py](./main1a_basic_text_loading.py)** - Simple text files and fundamental splitting
- **[main1b_csv_json_loading.py](./main1b_csv_json_loading.py)** - Structured data (CSV, JSON) handling  
- **[main1c_directory_batch_loading.py](./main1c_directory_batch_loading.py)** - Multiple files and batch processing
- **[main1d_advanced_splitting.py](./main1d_advanced_splitting.py)** - Code-aware and semantic splitting

```python
# Example: Basic document loading pattern
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("document.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

### 2️⃣ [main2_embeddings_vectorstore.py](./main2_embeddings_vectorstore.py)
**Embeddings and Vector Store Operations - Overview**

Complete guide to embeddings and vector stores with focused sub-modules:
- **[main2a_basic_embeddings.py](./main2a_basic_embeddings.py)** - Understanding embeddings and Google Gemini
- **[main2b_chroma_vectorstore.py](./main2b_chroma_vectorstore.py)** - Vector store operations and document management
- **[main2c_advanced_retrieval.py](./main2c_advanced_retrieval.py)** - Advanced search strategies and optimization

```python
# Example: Creating vector store with embeddings
vectorstore = FAISS.from_documents(documents, embeddings)
results = vectorstore.similarity_search("query", k=3)
```

### 3️⃣ [main3_basic_retriever.py](./main3_basic_retriever.py)
**Retriever Implementation - Overview**

Comprehensive guide to retriever setup with focused sub-modules:
- **[main3a_basic_retriever.py](./main3a_basic_retriever.py)** - Basic retriever setup and configuration
- **[main3b_retriever_agent.py](./main3b_retriever_agent.py)** - Using retrievers with LLM agents
- **[main3c_advanced_retrieval.py](./main3c_advanced_retrieval.py)** - Advanced strategies and error handling

```python
# Example: Basic retriever setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("your query here")
```

### 4️⃣ [main4_two_step_rag.py](./main4_two_step_rag.py)
**Two-Step RAG Implementation - Overview**

Traditional retrieve-then-generate RAG with focused sub-modules:
- **[main4a_basic_implementation.py](./main4a_basic_implementation.py)** - Core TwoStepRAGSystem and basic functionality
- **[main4b_context_engineering.py](./main4b_context_engineering.py)** - Advanced prompt templates and context optimization
- **[main4c_rag_evaluation.py](./main4c_rag_evaluation.py)** - Quality assessment and performance evaluation

```python
# Example: Basic two-step RAG pattern
class TwoStepRAGSystem:
    def answer_question(self, query: str, k: int = 3):
        # Step 1: Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)
        # Step 2: Generate answer using context
        context = "\n".join([doc.page_content for doc in docs])
        return self.model.invoke(self.rag_prompt.format_messages(context=context, question=query))
```

### 5️⃣ [main5_agentic_rag.py](./main5_agentic_rag.py)
**Agentic RAG Implementation**

Shows agents that decide when and how to retrieve:
- Create agents that reason about retrieval needs
- Implement dynamic retrieval strategies during reasoning
- Combine multiple retrieval tools for different data sources
- Handle complex multi-step research workflows

```python
# Example: Agent with multiple retrieval tools
agent = create_agent(
    model,
    tools=[
        search_technical_docs,
        search_product_info,
        search_web_info,
        calculate_pricing
    ],
    system_prompt="Use appropriate tools based on question type..."
)
```

### 6️⃣ [main6_hybrid_rag.py](./main6_hybrid_rag.py)
**Hybrid RAG Implementation**

Combines 2-step and agentic approaches with validation:
- Implement query enhancement and validation steps
- Handle iterative refinement and quality control
- Create adaptive RAG systems with fallback strategies
- Evaluate and improve retrieval quality

```python
# Example: Hybrid RAG with validation
class HybridRAGSystem:
    def process_query(self, query, max_iterations=2):
        for iteration in range(max_iterations):
            enhanced_query = self.query_enhancer.enhance_query(query)
            docs = self.retrieve_documents(enhanced_query)
            validation = self.validate_retrieval(query, docs)
            
            if validation["sufficient"]:
                answer = self.generate_answer(query, docs)
                return self.validate_answer(answer)
```

## 🎯 RAG Architectures

| Architecture | Description | Control | Flexibility | Latency | Example Use Case |
|-------------|-------------|---------|-------------|---------|-----------------|
| **2-Step RAG** | Retrieval always happens before generation | ✅ High | ❌ Low | ⚡ Fast | FAQs, documentation bots |
| **Agentic RAG** | Agent decides when and how to retrieve | ❌ Low | ✅ High | ⏳ Variable | Research assistants, multi-tool access |
| **Hybrid RAG** | Combines both approaches with validation | ⚖️ Medium | ⚖️ Medium | ⏳ Variable | Domain Q&A with quality control |

## 🔧 Retrieval Pipeline Components

### Document Loading
- **Text files**: PDF, Word, plain text, markdown
- **Web content**: HTML pages, APIs, RSS feeds  
- **Databases**: SQL, NoSQL, vector databases
- **Structured data**: JSON, XML, CSV

### Text Splitting
- **Recursive splitters**: Hierarchical splitting by separators
- **Token-based splitters**: Split by token count for LLM compatibility
- **Semantic splitters**: Split at semantic boundaries
- **Code-aware splitters**: Handle code and documentation

### Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small/large
- **HuggingFace**: sentence-transformers models
- **Local models**: Run embeddings locally for privacy
- **Domain-specific**: Fine-tuned for specific domains

### Vector Stores
- **FAISS**: Fast similarity search, good for prototyping
- **Chroma**: Persistent storage with metadata filtering
- **Pinecone**: Managed vector database service
- **Weaviate**: GraphQL-based vector search

### Retrievers
- **Vector retrieval**: Similarity-based document retrieval
- **BM25**: Traditional keyword-based retrieval
- **Hybrid**: Combine vector and keyword approaches
- **Multi-vector**: Different embeddings for same content

## 🎨 Design Patterns

### 1. Basic RAG Pattern
```python
# Load → Split → Embed → Store → Retrieve → Generate
documents = load_documents()
chunks = split_text(documents)
embeddings = create_embeddings(chunks)
vectorstore = store_embeddings(embeddings)
retriever = create_retriever(vectorstore)
```

### 2. Multi-Source Pattern
```python
# Different retrievers for different data sources
technical_retriever = create_retriever(technical_docs)
product_retriever = create_retriever(product_info)
web_retriever = create_web_retriever()
```

### 3. Validation Pattern
```python
# Quality control at each step
enhanced_query = enhance_query(original_query)
docs = retrieve_documents(enhanced_query)
quality_score = validate_retrieval(docs)
if quality_score > threshold:
    answer = generate_answer(docs)
```

### 4. Iterative Refinement
```python
# Improve results through multiple iterations
for attempt in range(max_attempts):
    result = try_retrieval_strategy(query, strategy=attempt)
    if meets_quality_threshold(result):
        return result
    query = refine_query(query, feedback=result.issues)
```

## 📊 Performance Optimization

### Chunk Size Guidelines
- **Small chunks (100-300 tokens)**: Better precision, may lack context
- **Medium chunks (300-800 tokens)**: Balanced approach for most use cases
- **Large chunks (800+ tokens)**: Better context, may include irrelevant info

### Retrieval Parameters
- **k (number of results)**: Start with 3-5, adjust based on context window
- **Score threshold**: Filter low-relevance results (typically 0.7+)
- **MMR (diversity)**: Balance relevance vs diversity in results

### Embedding Optimization
- **Model selection**: Balance accuracy vs speed/cost
- **Batch processing**: Process multiple documents together
- **Caching**: Store embeddings for frequently accessed content
- **Fine-tuning**: Adapt models to specific domains

## 🔍 Evaluation Metrics

### Retrieval Quality
- **Precision@K**: Fraction of retrieved docs that are relevant
- **Recall@K**: Fraction of relevant docs that are retrieved  
- **MRR**: Mean reciprocal rank of first relevant result
- **NDCG@K**: Normalized discounted cumulative gain

### Answer Quality
- **BLEU/ROUGE**: Text similarity to reference answers
- **BERT Score**: Semantic similarity using BERT embeddings
- **Human evaluation**: Expert assessment of answer quality
- **Factual accuracy**: Verification against ground truth

### System Performance
- **Latency**: Time from query to answer
- **Throughput**: Queries processed per second
- **Cost**: API calls and compute resources
- **Scalability**: Performance with increasing data size

## ⚖️ Trade-offs

**2-Step RAG**
- ✅ Predictable performance and cost
- ✅ Simple to implement and debug  
- ✅ Works well for straightforward Q&A
- ❌ May retrieve irrelevant information
- ❌ Limited flexibility for complex queries

**Agentic RAG**
- ✅ Intelligent retrieval decisions
- ✅ Handles complex, multi-step queries
- ✅ Can use multiple data sources
- ❌ Unpredictable latency and cost
- ❌ More complex to implement and debug

**Hybrid RAG**
- ✅ Quality validation and improvement
- ✅ Adaptive to different query types
- ✅ Good balance of control and flexibility
- ❌ Added complexity in implementation
- ❌ Higher latency due to validation steps

## 🎯 Choose Your Approach

1. **2-Step RAG** → Start here for most applications
2. **Agentic RAG** → When you need flexible, intelligent retrieval
3. **Hybrid RAG** → When quality control is critical
4. **Custom combinations** → Mix patterns based on specific needs

## 🛡️ Best Practices

1. **Data Quality**: Clean and structure your documents well
2. **Chunking Strategy**: Test different chunk sizes for your domain
3. **Embedding Selection**: Choose models appropriate for your content
4. **Retrieval Tuning**: Optimize parameters for your specific use case
5. **Quality Monitoring**: Implement evaluation and feedback loops
6. **Error Handling**: Gracefully handle retrieval and generation failures
7. **Privacy & Security**: Implement proper access controls and data protection