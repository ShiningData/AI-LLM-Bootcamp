#!/usr/bin/env python3
"""
HR RAG Chatbot - Main Application
=================================
A conversational HR assistant with short-term memory.

Usage:
    python main.py ingest              # Load documents into vector store
    python main.py chat                # Start interactive chat
    python main.py test                # Run predefined test questions
    python main.py update              # Update changed documents
    python main.py stats               # Show vector store statistics
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
import document_loader
import vector_store
import rag_agent


# Configuration
INITIAL_DOCS_DIR = "./hr_documents_pack/initial_docs"
UPDATED_DOCS_DIR = "./hr_documents_pack/updated_docs"
DEFAULT_MODEL = "openai:gpt-4o-mini"

# Test questions from assignment
TEST_QUESTIONS = [
    "What is the company's leave policy?",
    "How many vacation days do employees get?",
    "What are the steps in the offboarding process?",
    "What are the IT security requirements for new employees?",
    "What is the performance review process?",
    "How do I submit travel expenses for reimbursement?"
]


def cmd_ingest():
    """Load documents into the vector store."""
    print("\n" + "=" * 50)
    print("📥 DOCUMENT INGESTION")
    print("=" * 50)
    
    # Check if directory exists
    if not Path(INITIAL_DOCS_DIR).exists():
        print(f"❌ Directory not found: {INITIAL_DOCS_DIR}")
        print("Please add your HR documents first.")
        return
    
    # Process documents
    chunks = document_loader.process_documents(INITIAL_DOCS_DIR)
    
    # Add to vector store
    vector_store.add_documents(chunks)
    
    # Show stats
    stats = vector_store.get_stats()
    print(f"\n📊 Ingestion Complete!")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Files: {', '.join(stats['files'])}")


def cmd_update():
    """Update changed documents using hash comparison."""
    print("\n" + "=" * 50)
    print("🔄 DOCUMENT UPDATE")
    print("=" * 50)
    
    if not Path(UPDATED_DOCS_DIR).exists():
        print(f"❌ Directory not found: {UPDATED_DOCS_DIR}")
        return
    
    # Load updated documents
    documents = document_loader.load_documents(UPDATED_DOCS_DIR)
    
    updated = 0
    unchanged = 0
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        filename = Path(source).name
        
        # Calculate new hash
        new_hash = document_loader.calculate_file_hash(source)
        
        # Get existing hash from vector store
        existing_hash = vector_store.get_document_hash(filename)
        
        if existing_hash == new_hash:
            print(f"   ⏭️  {filename}: unchanged")
            unchanged += 1
        else:
            # Delete old version
            vector_store.delete_by_filename(filename)
            
            # Process and add new version
            chunks = document_loader.split_documents([doc])
            chunks = document_loader.add_metadata(chunks)
            vector_store.add_documents(chunks)
            
            print(f"   🔄 {filename}: updated")
            updated += 1
    
    print(f"\n📊 Update Complete!")
    print(f"   Updated: {updated}")
    print(f"   Unchanged: {unchanged}")


def cmd_chat():
    """Start interactive chat with memory."""
    print("\n" + "=" * 50)
    print("💬 HR RAG CHATBOT")
    print("=" * 50)
    print(f"Model: {DEFAULT_MODEL}")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Create agent and memory
    agent = rag_agent.create_hr_agent(DEFAULT_MODEL)
    memory = rag_agent.ConversationMemory()
    
    print("✅ Agent ready!\n")
    
    # Demo: Show memory feature
    print("💡 TIP: This chatbot has SHORT-TERM MEMORY!")
    print("   You can ask follow-up questions like:")
    print('   - "What about sick leave?" (after asking about leave)')
    print('   - "Can I carry them over?" (referring to previous topic)')
    print()
    
    while True:
        try:
            question = input("❓ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() == "quit":
            print("👋 Goodbye!")
            break
        
        # Get response with memory
        print("\n🔍 Searching...")
        answer = rag_agent.chat(agent, question, memory)
        print(f"\n🤖 Assistant: {answer}\n")


def cmd_test():
    """Run predefined test questions from assignment."""
    print("\n" + "=" * 50)
    print("🧪 RAG SYSTEM TEST")
    print("=" * 50)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Running {len(TEST_QUESTIONS)} test questions...")
    print("=" * 50)
    
    # Create agent (no memory for independent tests)
    agent = rag_agent.create_hr_agent(DEFAULT_MODEL)
    memory = rag_agent.ConversationMemory()
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 50}")
        print(f"📝 Question {i}/{len(TEST_QUESTIONS)}:")
        print(f"   {question}")
        print("─" * 50)
        
        try:
            # Clear memory for each question (independent tests)
            memory.clear()
            
            print("🔍 Searching...")
            answer = rag_agent.chat(agent, question, memory)
            
            print(f"\n🤖 Answer:")
            print(f"   {answer}")
            
            results.append({"question": question, "status": "✅ Success"})
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({"question": question, "status": f"❌ Error: {e}"})
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    success = sum(1 for r in results if "Success" in r["status"])
    print(f"Total: {len(results)} | Success: {success} | Failed: {len(results) - success}")
    
    for r in results:
        print(f"  {r['status']}: {r['question'][:40]}...")


def cmd_stats():
    """Show vector store statistics."""
    print("\n" + "=" * 50)
    print("📊 VECTOR STORE STATISTICS")
    print("=" * 50)
    
    stats = vector_store.get_stats()
    
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique files: {stats['unique_files']}")
    print(f"Files:")
    for f in stats['files']:
        print(f"  - {f}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "ingest":
        cmd_ingest()
    elif command == "update":
        cmd_update()
    elif command == "chat":
        cmd_chat()
    elif command == "test":
        cmd_test()
    elif command == "stats":
        cmd_stats()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()