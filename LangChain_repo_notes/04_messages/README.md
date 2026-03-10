## Messages
- Messages are the **fundamental unit of context** for models in LangChain. They represent the input and output of models, carrying both the content and metadata needed to represent the state of a conversation when interacting with an LLM. LangChain provides a standard message type that works across all model providers, ensuring consistent behavior regardless of the model being called.

- Messages are objects that contain:
 - **Role** - Identifies the message type (e.g. system, user)
 - **Content** - Represents the actual content of the message (like text, images, audio, documents, etc.)
 - **Metadata** - Optional fields such as response information, message IDs, and token usage
