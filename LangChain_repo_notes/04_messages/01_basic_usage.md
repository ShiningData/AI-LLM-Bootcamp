## Basic usage
- The simplest way to use messages is to create message objects and pass them to a model when invoking.
```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", max_tokens=200)

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage
print(response)
```
### Text prompts
- Text prompts are strings - ideal for straightforward generation tasks where you don’t need to retain conversation history.
```python
response = model.invoke("Write a haiku about spring")
```
- Use text prompts when:
    - You have a single, standalone request
    - You don’t need conversation history
    - You want minimal code complexity

### Message prompts
- Alternatively, you can pass in a list of messages to the model by providing a list of message objects.
```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```
- Use message prompts when:
    - Managing multi-turn conversations
    - Working with multimodal content (images, audio, files)
    - Including system instructions

### Dictionary format
- You can also specify messages directly in OpenAI chat completions format.
```python
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```
