## Rate limiting
- Many chat model providers impose a limit on the number of invocations that can be made in a given time period. If you hit a rate limit, you will typically receive a rate limit error response from the provider, and will need to wait before making more requests.
To help manage rate limits, chat model integrations accept a **rate_limiter** parameter that can be provided during initialization to control the rate at which requests are made.

## Token usage
- A number of model providers return token usage information as part of the invocation response. When available, this information will be included on the AIMessage objects produced by the corresponding model. For more details, see the messages guide.

```python
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
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/02_models/main10_token_usage.py
```
- Output
```
{'gemini-2.5-flash-lite': {'input_tokens': 2, 'output_tokens': 9, 'total_tokens': 11, 'input_token_details': {'cache_read': 0}}, 'gemini-2.5-flash': {'input_tokens': 2, 'output_tokens': 41, 'total_tokens': 43, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 31}}}
```