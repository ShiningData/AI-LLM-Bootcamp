# HR RAG Chatbot with Short-Term Memory

A simple RAG (Retrieval-Augmented Generation) chatbot that answers questions about HR documents. Built with LangChain v1.2.0.


**Key Features:**
- **Short-Term Memory**: Remembers conversation context for follow-up questions
- **Hash-Based Updates**: Detects document changes using SHA256 hashing
- **14 Metadata Fields**: Comprehensive document tracking per assignment specs
- **DirectoryLoader**: Simple document loading with LangChain

---

## 📁 Project Structure

```
hr_rag_chatbot/
├── document_loader.py    # Load and chunk documents (DirectoryLoader)
├── vector_store.py       # ChromaDB operations (collection: vbo-aillm-bc-rag)
├── rag_agent.py          # Agent with short-term memory
├── main.py               # CLI application
├── requirements.txt
├── .env.example
└── hr_documents_pack/
    ├── initial_docs/     # Your HR documents
    └── updated_docs/     # Updated versions
```

**Only 4 Python files!** Simple and easy to understand.

---

## 🚀 Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Add Documents

Put your HR documents in `hr_documents_pack/initial_docs/`:

```
hr_documents_pack/
└── initial_docs/
    ├── employee_handbook.docx
    ├── leave_policy.docx
    ├── it_security.pdf
    └── ...
```

### 3. Run

```bash
# Step 1: Load documents into vector store
python main.py ingest

# Step 2: Run test questions
python main.py test

# Step 3: Start interactive chat
python main.py chat
```

---

## 🧪 Test Questions

The system includes 6 predefined test questions from the assignment:

1. "What is the company's leave policy?"
2. "How many vacation days do employees get?"
3. "What are the steps in the offboarding process?"
4. "What are the IT security requirements for new employees?"
5. "What is the performance review process?"
6. "How do I submit travel expenses for reimbursement?"

Run with: `python main.py test`

---

## 🧠 Short-Term Memory

Run with: `python main.py chat`

The chatbot remembers your conversation:

```
❓ You: my name is alican

🔍 Searching...

🤖 Assistant: Nice to meet you, Alican! How can I assist you today?

❓ You: what is my name

🔍 Searching...

🤖 Assistant: Your name is Alican.

❓ You: What is the leave policy?

🔍 Searching...

🤖 Assistant: The leave policy includes the following:

- **Annual Leave**: 20 paid vacation days, with requests made 5 days in advance via the HR Portal.
- **Sick Leave**: 10 days per year, requiring a doctor's report if taken for more than 2 days.
- **Parental Leave**: Maternity leave is 16 weeks paid, paternity leave is 2 weeks paid, and up to 12 weeks unpaid parental leave is available.
- **Emergency Leave**: 3 paid days per year. 

(Source: leave_policy.docx)

❓ You: What about sick leave?

🔍 Searching...

🤖 Assistant: Sick leave allows for 10 days per year, and if more than 2 consecutive days are taken, a doctor's report is required. This information is part of the general leave policy.

```

---

## 📊 Metadata Fields (14 Total)

Each document chunk includes comprehensive metadata:

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

## 🔄 Updating Documents

When HR policies change:

1. Put updated files in `hr_documents_pack/updated_docs/`
2. Run: `python main.py update`

The system compares file hashes:
- Same hash → Skip (no changes)
- Different hash → Delete old, add new

---

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `python main.py ingest` | Load documents into vector store |
| `python main.py test` | Run predefined test questions |
| `python main.py chat` | Start interactive chat |
| `python main.py update` | Update changed documents |
| `python main.py stats` | Show vector store statistics |

