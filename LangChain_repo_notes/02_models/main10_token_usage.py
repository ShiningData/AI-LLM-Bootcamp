from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="google_genai:gemini-2.5-flash-lite")
model_2 = init_chat_model(model="google_genai:gemini-2.5-flash")

with get_usage_metadata_callback() as cb:
    model_1.invoke("Hello")
    model_2.invoke("Hello")
    print(cb.usage_metadata)

"""
{'gemini-2.5-flash-lite': {'input_tokens': 2, 'output_tokens': 9, 'total_tokens': 11, 'input_token_details': {'cache_read': 0}}, 'gemini-2.5-flash': {'input_tokens': 2, 'output_tokens': 41, 'total_tokens': 43, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 31}}}
"""