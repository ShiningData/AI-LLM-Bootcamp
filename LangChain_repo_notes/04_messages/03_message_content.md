
## Message content
- You can think of a message’s content as the payload of data that gets sent to the model. Messages have a content attribute that is loosely-typed, supporting strings and lists of untyped objects (e.g., dictionaries). This allows support for provider-native structures directly in LangChain chat models, such as multimodal content and other data.
- LangChain chat models accept message content in the content attribute, and can contain:
    - A string
    - A list of content blocks in a provider-native format
    - A list of LangChain’s standard content blocks
See below for an example using multimodal inputs:
```python
from langchain.messages import HumanMessage

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider-native format (e.g., OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```