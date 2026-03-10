
## Runnable

In modern LangChain (v1 architecture), **Runnable** is the **core building block**.
Everything—LLMs, prompts, retrievers, embeddings, functions, sequences—is treated as a **Runnable**.

Think of it as:

> A *unified interface* that lets you compose, chain, parallelize, and transform operations in an LLM pipeline.

It solves the biggest pain point of old LangChain:
different objects had different `.run`, `.invoke`, `.call`, `.predict` methods.

Now everything has **one standard interface**:

* `invoke()` → single input, single output
* `batch()` → multiple inputs in parallel
* `stream()` → token streaming

---

## Why is Runnable Important?

Because it lets you build **modular**, **testable**, **composable** LLM pipelines—like LEGO blocks.

You can treat anything as a step:

* PromptTemplate
* LLM call
* Tools
* Retrievers
* Embedding models
* Custom Python functions
* VectorDB queries
* Message history
* Whole RAG pipelines

All unified under **Runnable**.

---

# Runnable Mindset (simple analogy)

Imagine:

* A *Prompt* is a block.
* A *Model* is a block.
* A *Retriever* is a block.
* A *JSON parser* is a block.

A **Runnable** is the *pipe* that connects these blocks so data flows correctly.

---

## Minimal Runnable Example
Using:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
```

### 1. Create building blocks (everything is a Runnable)

```python
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

### 2. Compose them with RunnablePipe (`|`)

```python
chain = prompt | llm
```

### 3. Run them

```python
response = chain.invoke({"topic": "embeddings"})
print(response.content)
```

This works because both prompt and llm are **Runnable** objects.

---

# Runnable Types

### 1. **RunnableSequence**

Order of operations:

```python
chain = prompt | llm | StrOutputParser()
```

### 2. **RunnableMap**

Run things in parallel:

```python
chain = {
    "summary": summarizer,
    "keywords": keyword_extractor
}
```

### 3. **RunnableLambda**

Custom Python functions inside chains:

```python
def upper(x): return x.upper()
chain = RunnableLambda(upper)
```

### 4. **RunnableParallel**

Parallel tasks with separate logic.

### 5. **RunnablePassthrough**

Use the input itself in your chain.

---


### 6. Summary

| Concept            | Meaning                               |                           |
| ------------------ | ------------------------------------- | ------------------------- |
| **Runnable**       | Universal operation unit in LangChain |                           |
| **invoke**         | Run once                              |                           |
| **batch**          | Run in parallel                       |                           |
| **stream**         | Token-by-token output                 |                           |
| **RunnableMap**    | Parallel steps                        |                           |
| **RunnableLambda** | Custom python step                    |                           |

---

### 7. All examples
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/02_models/main11_runnable.py
```
