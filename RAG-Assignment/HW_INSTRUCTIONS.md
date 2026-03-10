# Week 6 Homework: RAG Chatbot with Short-Term Memory

## 🎯 Objective

Build a RAG (Retrieval-Augmented Generation) chatbot that:
1. Answers questions about HR documents
2. Remembers conversation context (short-term memory)
3. Updates documents when they change (hash-based detection)

---

## 📋 Requirements

### Technical Requirements

- Python >= 3.10
- LangChain >= 1.2.0
- Use `create_agent` from `langchain.agents` (not deprecated methods)
- Use `DirectoryLoader` for loading documents
- Use ChromaDB for vector storage
- **Collection Name:** `vbo-aillm-bc-rag`

### Functional Requirements

| Feature | Description |
|---------|-------------|
| Document Ingestion | Load DOCX, PDF, TXT files and store in vector database |
| RAG Query | Answer questions using retrieved document context |
| Short-Term Memory | Remember conversation for follow-up questions |
| Document Update | Detect and update changed documents using hash comparison |

---

## 📁 Expected Structure

```
hr_rag_chatbot/
├── document_loader.py    # Document loading and processing
├── vector_store.py       # ChromaDB operations
├── rag_agent.py          # Agent with memory
├── main.py               # CLI application
├── requirements.txt
├── .env.example
├── README.md
└── hr_documents_pack/
    ├── initial_docs/     # Original documents
    └── updated_docs/     # Updated documents for testing
```

---

## 📝 Tasks

### Task 1: Document Loading (20 points)

Use `DirectoryLoader` to load documents:

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path="hr_documents_pack/initial_docs/")
documents = loader.load()
```

Requirements:
- Load all supported file types (DOCX, PDF, TXT)
- Split into chunks (500 chars, 100 overlap)
- Add 14 metadata fields (see Metadata section below)

### Task 2: Vector Store (20 points)

Implement ChromaDB operations:

```python
from langchain_chroma import Chroma

store = Chroma(collection_name="vbo-aillm-bc-rag")
```

Required functions:
- `add_documents(documents)` - Add to store
- `search(query, k=4)` - Similarity search
- `delete_by_filename(filename)` - Delete by metadata
- `get_document_hash(filename)` - Get stored hash

### Task 3: Short-Term Memory (25 points)

Implement conversation memory:

```python
class ConversationMemory:
    def __init__(self, max_messages=10):
        self.messages = []
    
    def add_user_message(self, content): ...
    def add_ai_message(self, content): ...
    def get_messages(self): ...
```

The agent should:
- Remember previous questions and answers
- Understand references ("it", "them", "that policy")

### Task 4: RAG Agent (25 points)

Create agent using LangChain v1.2.0:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_hr_documents(query: str):
    """Search HR documents for relevant information."""
    ...

agent = create_agent(
    model=model,
    tools=[search_hr_documents],
    system_prompt="..."
)
```

Requirements:
- Use `@tool` decorator for retrieval
- Keep answers SHORT (2-3 sentences)
- Always cite source documents

### Task 5: Document Update (10 points)

Implement hash-based update:

```python
def update_document(filename):
    new_hash = calculate_hash(new_file)
    old_hash = get_stored_hash(filename)
    
    if new_hash != old_hash:
        delete_old_chunks(filename)
        add_new_chunks(filename)
```

---

## 📊 Required Metadata (14 Fields)

Each chunk must include these metadata fields:

| Field | Description |
|-------|-------------|
| `file_name` | Original filename for update tracking |
| `file_extension` | File extension (.docx, .pdf, .txt) |
| `file_size_bytes` | Original file size in bytes |
| `character_count` | Total character count of document |
| `chunk_index` | Position within the document |
| `chunk_size` | Size of current chunk in characters |
| `chunk_overlap` | Overlap size used during chunking |
| `document_type` | Format category (document, text, pdf) |
| `creation_date` | File creation timestamp |
| `last_modified` | File last modified timestamp |
| `ingestion_timestamp` | When ingested into system |
| `document_hash` | SHA256 hash for change detection |
| `page_number` | Page number (for PDFs) |
| `section_title` | Section heading if available |

---

## 🧪 Testing

### Test Questions (Required)

Run these questions with `python main.py test`:

1. "What is the company's leave policy?"
2. "How many vacation days do employees get?"
3. "What are the steps in the offboarding process?"
4. "What are the IT security requirements for new employees?"
5. "What is the performance review process?"
6. "How do I submit travel expenses for reimbursement?"

### Short-Term Memory Test

```
You: What is the leave policy?
Bot: Employees get 20 vacation days...

You: What about sick leave?
Bot: [Should understand context and answer about sick leave]

```

### Document Update Test

```
python main.py update

# Should show:
# - employee_handbook.docx: updated
# - other_file.docx: unchanged
```

---

## 📊 Grading

| Task | Points |
|------|--------|
| Document Loading (14 metadata) | 20 |
| Vector Store | 20 |
| Short-Term Memory | 25 |
| RAG Agent | 25 |
| Document Update | 10 |
| **Total** | **100** |

