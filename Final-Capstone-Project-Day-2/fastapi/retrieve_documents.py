from langchain_qdrant import Qdrant, QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # 1. Set the API Key (starts with 'sk-or-...')
    openai_api_key=os.getenv('OPENROUTER_API_KEY'),
    # 2. Override the Base URL to point to OpenRouter
    openai_api_base="https://openrouter.ai/api/v1"
)

url="http://vectordb:6333"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="my_documents",
    url=url,
)


if __name__=='__main__':
    result = qdrant.similarity_search(query="Hadoop dersi var mı?")
    print(result)