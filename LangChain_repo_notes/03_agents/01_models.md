## 1. What is the difference between model.bind_tools and Agent
- The difference is that an **Agent** is a system that runs in a loop, whereas `model.bind_tools` is just a single step.

### 1.1. The Raw Model (`bind_tools`)

When you use `model.bind_tools(tools).invoke(...)`, you are making a **single API call**.

* The model looks at your question.
* It looks at the tool definitions.
* It decides: "I need to call the `get_weather` tool with `location=Boston`."
* **It stops there.** It returns the "Tool Call" request to you. It cannot execute your Python code, so it doesn't know the result (It is sunny) yet.

### 1.2. The Agent (`create_agent`)

The agent wrapper (like the one you used from LangGraph/LangChain) creates a **control loop** that automates the back-and-forth for you.

Here is what happens behind the scenes when you run `agent.invoke`:

| Step | Actor | Action |
| --- | --- | --- |
| **1** | **LLM** | Sees the prompt and outputs a **Tool Call** (e.g., `get_weather(location='Boston')`). |
| **2** | **Agent Loop** | Intercepts that output, sees it's a tool call, and **actually executes** your Python function. |
| **3** | **Agent Loop** | Takes the result (`It is sunny`) and sends it **back** to the LLM as a new message. |
| **4** | **LLM** | Sees the original question + the tool result, and then generates the final text: "The answer is The weather in Boston is sunny." |

---

### 1.3. Why didn't `bind_tools` work for the weather?

In your previous weather example, you were calling the model **once**. It successfully generated the "instruction" to check the weather in Boston, but there was no "Agent Loop" to:

1. See that instruction.
2. Run your `get_weather` function.
3. Give the answer back to the model for a final summary.

### 1.4. Summary of terminology

* **Tool Calling:** The LLM's ability to output structured JSON instead of plain text.
* **Tool Execution:** The act of running the actual Python code (LLMs cannot do this).
* **Agent:** The "engine" that connects the two by passing data back and forth until the task is done.

----

## 2. Model
- The model is the reasoning engine of your agent. It can be specified in multiple ways, supporting both static and dynamic model selection.

### 2.1. Static Model
A static model configuration uses a single, pre-defined model throughout the entire agent lifecycle. This approach is straightforward and suitable for most use cases where consistent model behavior is desired. The model is initialized once with specific parameters and remains unchanged during execution.

- main1_create_agent.py
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.1,
    max_tokens=500,
)

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two integers"""
    return x * y

tools = [multiply]

agent = create_agent(
    model=model,
    tools=tools
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 5 times 7?"}]})
print("Final answer:", result["messages"][-1].content)
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/03_agents/main1_create_agent.py
```

- For model strings and inference list: https://reference.langchain.com/python/langchain/models/

### 2.2. Dynamic model
Dynamic model selection allows agents to switch between different models based on runtime conditions such as conversation complexity, message length, or specific requirements. This approach optimizes performance and cost by using simpler models for basic tasks and more powerful models for complex reasoning, enabling adaptive behavior throughout the conversation.

- main2_dynamic_model_usage.py
```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
    system_prompt="You are a helpful assistant. Be concise and accurate."
    # ... (other params)
)

advanced_model = ChatOpenAI(
    model="gpt-4o"
    # ... (other params)
    )

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```
