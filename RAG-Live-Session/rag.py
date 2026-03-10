from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from ingestion import vector_store

load_dotenv()

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