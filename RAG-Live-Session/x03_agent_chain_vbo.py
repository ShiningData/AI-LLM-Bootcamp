# Chroma requires pysqlite3 instead of sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# Web loading
# Only keep post title, headers, and content from the full HTML.

docs = DirectoryLoader(
   path="./text_files"
).load()


# Split documents into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=100,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

all_splits = text_splitter.split_documents(docs)


embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )


vector_store = Chroma(
    collection_name="vbo-bootcamps",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Ingest chunks into vectore store
document_ids = vector_store.add_documents(documents=all_splits)

model = init_chat_model("google_genai:gemini-2.5-flash")


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=4)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "Sen bir eğitim kurumunun yardımsever bir satış asistanısın. Cevabını aşağıdaki bilgiyi kullanarak ver:"
        f"\n\n{docs_content}"
    )

    print(f"Printing system_message in dynamic-prompt: \n\n {system_message}")

    return system_message

agent = create_agent(
    model=model,
    tools=[],
    middleware=[prompt_with_context]
)




if __name__=='__main__':
    query = "Apache Spark dersi var mı? Varsa hangi konular işleniyor? "

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        ):
        event["messages"][-1].pretty_print()
