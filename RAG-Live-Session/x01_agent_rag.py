"""
Docstring for week_06_rag.00_1_rag_example.main
Official langchain RAG example
https://docs.langchain.com/oss/python/langchain/rag
"""
# Chroma requires pysqlite3 instead of sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# Web loading
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

web_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

web_docs = web_loader.load()

# # Text loading
# text_loader = TextLoader(file_path="./text_files/text_file_1.txt")

# text_docs = text_loader.load()

# # Pdf loading
# pdf_loader = PyPDFLoader(file_path="./chroma_langchain_db/pdf_files/fronend-test.pdf")

# pdf_docs = pdf_loader.load()

# # Merge all documents
# all_docs = web_docs + text_docs + pdf_docs


# Split documents into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

all_splits = text_splitter.split_documents(web_docs)


embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Ingest chunks into vectore store
document_ids = vector_store.add_documents(documents=all_splits)

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=prompt
)

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)


if __name__=='__main__':
    # print("docs type:", type(web_docs))
    # print(f"Length of docs: {len(web_docs)}")
    # print(f"Type of each list element of web content: {type(web_docs[0])}")
    # # print(f"Content of first document: \n\n {web_docs[0]}")

    # print("text loader ===================================")
    # print("docs type:", type(text_docs))
    # print(f"Length of docs: {len(text_docs)}")
    # print(f"Type of each list element of text content: {type(text_docs[0])}")

    # print("pdf loader ===================================")
    # print("docs type:", type(pdf_docs))
    # print(f"Length of docs: {len(pdf_docs)}")
    # print(f"Type of each list element of pdf content: {type(pdf_docs[0])}")
    # print("All docs ===================================")
    # print("docs type:", type(all_docs))
    # print(f"Length of docs: {len(all_docs)}")
    # print(f"Type of each list element of pdf content: {type(all_docs[0])}")

    # print("docs type:", type(all_splits))
    # print(f"Length of docs: {len(all_splits)}")
    # print(f"Type of each list element of chunk/splitted content: {type(all_splits[0])}")

    # for  i, doc in enumerate(all_splits[:3], 1):
    #     print(f"Doc number: {i} {len(doc.page_content)} \n\n")
    #     print(doc)

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        ):
        event["messages"][-1].pretty_print()
