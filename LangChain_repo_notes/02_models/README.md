## Models
- LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate languages, summarize, and answer questions without needing specialized training for each task.
- Models are the reasoning engine of agents. They drive the agent’s decision-making process, determining which tools to call, how to interpret results, and when to provide a final answer.
- LangChain’s standard model interfaces give you access to many different provider integrations, which makes it easy to experiment with and switch between models to find the best fit for your case.

- In addition to text generation, many models support:
    - **Tool calling** - calling external tools (like databases queries or API calls) and use results in their responses.
    - **Structured output** - where the model’s response is constrained to follow a defined format.
    - **Multimodality** - process and return data other than text, such as images, audio, and video.
    - **Reasoning** - models perform multi-step reasoning to arrive at a conclusion.

### Basic usage
- Models can be utilized in two ways:
    - With agents - Models can be dynamically specified when creating an agent.
    - Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, classification, or extraction without the need for an agent framework.

### Initialize a model
- main.py
```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    # max_retries, api_key

    )

response = model.invoke("Why do parrots talk?")
print(response)
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/02_models/main.py
```


