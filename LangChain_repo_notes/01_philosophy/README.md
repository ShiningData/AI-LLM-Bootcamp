# LangChain Philosophy


## Why LangChain?

* Large Language Models (LLMs) on their own are powerful but **not enough** on their own.
  They often lack access to external data, the ability to call tools, or to keep track of multi-step workflows.
* LangChain fills the gap: it helps you build **real-world AI applications** (not just single prompts) by combining LLMs + data + tools + workflows.
* In short: *“turn raw LLM capability into useful apps.”*

## Core Principles

1. **Modularity**

   * LangChain breaks down your AI system into reusable building blocks: models, prompts, tools, memory, retrieval, etc.
   * This means you can swap components (e.g., a different model) without rewriting everything.

2. **Composability / Orchestration**

   * You can connect building blocks into pipelines and workflows ("chains", "agents", etc.).
   * For example: load documents → embed them → retrieve relevant parts → ask model → parse result.

3. **Standardised Abstractions**

   * LangChain provides uniform interfaces for different models, memory systems, tools, vector stores, etc.
   * You don’t have to learn a completely new API per model or vendor.

4. **Production-ready focus**

   * While prototyping is fine, LangChain keeps an eye on what it takes to build scalable, reliable AI systems: retrieval, memory, tool calling, monitoring, etc.
   * It supports features like streaming, batching, memory, tooling, etc.

## Key Concepts (in simple terms)

* **Chat models & Messages**: The “conversation” interface—user messages, AI messages.
* **Tools**: Functions that the model can call (e.g., search, calculate, send email) with a schema (name, args).
* **Memory**: The system’s way of remembering past interactions to provide context.
* **Retrieval / Embeddings / Vector stores**: When you have a lot of data (documents) and you want the model to use relevant parts of it.
* **Agents**: Smart systems that use a model + tools + retrieval + memory to decide what to do and then do it.
* **Structured Output**: Rather than free-text, the system can return more structured data (JSON, typed objects) to make it reliable and parsable.
