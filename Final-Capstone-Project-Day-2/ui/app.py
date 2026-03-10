import streamlit as st
import requests
import pandas as pd
import json

# Configure Streamlit page
st.set_page_config(
    page_title="SQL Database Query Interface", 
    page_icon="🗄️",
    layout="wide"
)

# Title and description
st.title("🗄️ SQL Database Query Interface")
st.markdown("Ask natural language questions about your database and get results in a clean table format.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to call FastAPI backend
def query_database(question: str):
    """Send question to FastAPI backend and return response"""
    try:
        fastapi_url = "http://fastapi:8000/chat"  # Docker service name
        payload = {"message": question}
        
        response = requests.post(fastapi_url, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Function to parse and display results
def display_results(response_data):
    """Parse and display the database query results"""
    if "error" in response_data:
        st.error(f"❌ Error: {response_data['error']}")
        return
    
    if "response" not in response_data:
        st.error("❌ Invalid response format from backend")
        return
    
    response_content = response_data["response"]
    
    # Try to extract table data from the response
    if isinstance(response_content, str):
        # Look for structured data patterns
        if "```" in response_content:
            # Extract code blocks that might contain table data
            code_blocks = response_content.split("```")
            for i, block in enumerate(code_blocks):
                if i % 2 == 1:  # Odd indices are code blocks
                    st.code(block, language="sql")
        
        # Display the full response
        st.markdown("### 📊 Query Result:")
        st.markdown(response_content)
        
        # Try to extract tabular data if present
        lines = response_content.split('\n')
        table_data = []
        headers = None
        
        for line in lines:
            if '|' in line and line.count('|') > 1:
                # This looks like a table row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    if headers is None:
                        headers = cells
                    elif cells != headers and not all(cell in ['-', '---', '----'] for cell in cells):
                        table_data.append(cells)
        
        # Display as DataFrame if we found tabular data
        if headers and table_data:
            try:
                df = pd.DataFrame(table_data, columns=headers)
                st.markdown("### 📋 Results Table:")
                st.dataframe(df, use_container_width=True)
            except Exception:
                pass  # If DataFrame creation fails, just show the text
    
    else:
        st.json(response_content)

# Main interface
with st.container():
    # Input section
    st.markdown("### 💬 Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Enter your database question:",
            placeholder="e.g., Show me all customers from last month",
            key="question_input"
        )
    
    with col2:
        submit_button = st.button("🔍 Query", type="primary")

    # Process query
    if submit_button and user_question.strip():
        # Add user question to chat history
        st.session_state.chat_history.append({"type": "user", "content": user_question})
        
        # Show loading spinner
        with st.spinner("🔄 Querying database..."):
            response = query_database(user_question)
        
        # Add response to chat history
        st.session_state.chat_history.append({"type": "assistant", "content": response})
        
        # Display current result
        display_results(response)

# Chat history section
if st.session_state.chat_history:
    st.markdown("### 📝 Query History")
    
    # Show chat history in reverse order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        if chat["type"] == "user":
            with st.expander(f"❓ Question {len(st.session_state.chat_history)//2 - i//2}: {chat['content'][:50]}..."):
                st.markdown(f"**Question:** {chat['content']}")
        elif chat["type"] == "assistant":
            with st.expander(f"📊 Answer {len(st.session_state.chat_history)//2 - i//2}"):
                display_results(chat["content"])

# Clear history button
if st.session_state.chat_history:
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Sidebar with examples and info
with st.sidebar:
    st.markdown("### 💡 Example Questions")
    examples = [
        "Show me all tables in the database",
        "List the first 5 customers",
        "What are the column names in the users table?",
        "Count the total number of orders",
        "Show me recent transactions"
    ]
    
    for example in examples:
        if st.button(f"📋 {example}", key=f"example_{hash(example)}"):
            st.session_state.question_input = example
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This interface allows you to query your PostgreSQL database using natural language.
    
    - Ask questions in plain English
    - Get structured results in table format
    - View query history
    - See SQL queries generated
    """)
    
    st.markdown("### 🔧 Status")
    # Simple health check
    try:
        health_response = requests.get("http://fastapi:8000", timeout=5)
        st.success("✅ FastAPI Backend: Connected")
    except:
        st.error("❌ FastAPI Backend: Disconnected")