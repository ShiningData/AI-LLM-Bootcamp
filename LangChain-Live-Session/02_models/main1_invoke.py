from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

result = model.invoke("Ankara'da bugün hava nasıl? Bugün ne?")

print(result)