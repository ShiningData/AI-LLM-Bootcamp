
## Structured output
In some situations, you may want the agent to return an output in a specific format. LangChain provides strategies for structured output via the response_format parameter.

### ToolStrategy
ToolStrategy uses artificial tool calling to generate structured output. This works with any model that supports tool calling:

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main5_structured_output.py
```
### ProviderStrategy
ProviderStrategy uses the model provider’s native structured output generation. This is more reliable but only works with providers that support native structured output (e.g., OpenAI):
```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```
- As of langchain 1.0, simply passing a schema (e.g., response_format=ContactInfo) is no longer supported. You must explicitly use ToolStrategy or ProviderStrategy.