## LangChain overview
- LangChain is the easiest way to start building agents and applications powered by LLMs.

- LangChain provides;
    - Pre-built agent architecture
    - Model integrations 
    - To help get started quickly and seamlessly incorporate LLMs into agents and applications.

- LangGraph, low-level agent orchestration framework and runtime, for more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.


- LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

- Requires Python 3.10+

## What's new in v1
Here’s a **Markdown** summary of the release notes for **LangChain v1.0 (Python)** (based on the official docs). ([LangChain Docs][1])

---

## LangChain v1.0 — What’s New

### Focus

LangChain v1 is positioned as a **production-ready, streamlined foundation** for building agents. ([LangChain Docs][1])
The release centres on three main improvements:

1. A new **`create_agent`** API. ([LangChain Docs][1])
2. **Standard content blocks** for model output (across providers) via a unified API. ([LangChain Docs][1])
3. A **simplified package namespace**, where legacy functionality is moved into a separate package (`langchain-classic`). ([LangChain Docs][1])

---

## Key Features & Changes

### `create_agent`

* The new standard for building agents. ([LangChain Docs][1])
* Example usage:

  ```python
  from langchain.agents import create_agent

  agent = create_agent(
      model="claude-sonnet-4-5-20250929",
      tools=[search_web, analyze_data, send_email],
      system_prompt="You are a helpful research assistant."
  )

  result = agent.invoke({
      "messages": [
          {"role": "user", "content": "Research AI safety trends"}
      ]
  })
  ```

  ([LangChain Docs][1])
* Under-the-hood: it follows the basic agent loop (model call → tool calls → finish when no tool call) and is built on top of LangGraph. ([LangChain Docs][1])

### Middleware

* Agents via `create_agent` support a **middleware** system: you can plug into hooks to manipulate behaviour before/after model calls, before/after tool calls etc. ([LangChain Docs][1])
* Prebuilt middleware examples:

  * `PIIMiddleware` — redact sensitive info
  * `SummarizationMiddleware` — condense conversation history
  * `HumanInTheLoopMiddleware` — require human approval for sensitive tool calls ([LangChain Docs][1])
* You can also implement **custom middleware** by subclassing `AgentMiddleware` and using hooks like `wrap_model_call`, `before_agent`, `after_agent`, etc. ([LangChain Docs][1])

### Structured Output

* `create_agent` integrates structured output via `ToolStrategy` or other schema definitions (e.g., Pydantic models) so that model responses can be parsed into typed output. ([LangChain Docs][1])

* Example:

  ```python
  from langchain.agents.structured_output import ToolStrategy
  from pydantic import BaseModel

  class Weather(BaseModel):
      temperature: float
      condition: str

  def weather_tool(city: str) -> str:
      return f"it's sunny and 70 degrees in {city}"

  agent = create_agent(
      "gpt-4o-mini",
      tools=[weather_tool],
      response_format=ToolStrategy(Weather)
  )
  result = agent.invoke({
      "messages": [{"role": "user", "content": "What's the weather in SF?"}]
  })
  print(repr(result["structured_response"]))
  # → Weather(temperature=70.0, condition='sunny')
  ```

  ([LangChain Docs][1])

* Error handling for structured output (parsing errors, multiple tool calls when only one expected) is configurable via `handle_errors` parameter. ([LangChain Docs][1])

### Standard Content Blocks

* Model responses from different providers now expose a unified `.content_blocks` API: allows you to inspect reasoning traces, tool calls, citations, etc., independent of provider. ([LangChain Docs][1])
* Example with the `content_blocks` property:

  ```python
  for block in response.content_blocks:
      if block["type"] == "reasoning":
          print(f"Model reasoning: {block['reasoning']}")
      elif block["type"] == "text":
          print(f"Response: {block['text']}")
      elif block["type"] == "tool_call":
          print(f"Tool call: {block['name']}({block['args']})")
  ```

  ([LangChain Docs][1])
* Benefits:

  * Provider-agnostic access to reasoning/tool calls/citations.
  * Type-safe (with type hints) for content block types.
  * Backward compatibility: standard content blocks load lazily so minimal disruption. ([LangChain Docs][1])

### Simplified Package & Namespace

* The top-level `langchain` namespace now focuses on essential building blocks: agents, models, messages, tools, embeddings, etc. ([LangChain Docs][1])
* The legacy abstractions (chains, older retrievers, indexing API, hub module, etc.) have been moved to a separate package `langchain-classic`. ([LangChain Docs][1])
* The example of new API surface:

  ```python
  from langchain.agents import create_agent
  from langchain.messages import AIMessage, HumanMessage
  from langchain.tools import tool
  from langchain.chat_models import init_chat_model
  from langchain.embeddings import init_embeddings
  ```

  ([LangChain Docs][1])

---

## Migration & Compatibility

* A comprehensive migration guide is available—needed if you are migrating from v0.x to v1.0. ([LangChain Docs][2])
* Legacy patterns still supported via `langchain-classic`, so older code (chains, retrievers, indexing) can continue to work with minimal disruption. ([LangChain Docs][1])
* Note: While not explicitly in the release notes for LangChain (but relevant via LangGraph), Python 3.9 support is dropped—now Python 3.10+ is required. ([LangChain Docs][3])

---

## Why This Matters

* According to external commentary, the redesign addresses complexity that had built up: chains, agents, tools, wrappers, prompt helpers had proliferated and made the API surface fragmented. ([TECHCOMMUNITY.MICROSOFT.COM][4])
* The tighter agent abstraction (the “tool-calling agent loop”) and dependency on LangGraph internals means better reliability, more consistent workflows for production systems. ([LangChain Blog][5])
* For developers building LLM-agent systems (which matches your focus in bootcamp), the simplified interface + richer orchestration (middleware, structured output, content blocks) means less boilerplate and more robust architecture.

---

## Summary Table

| Area               | Change / Improvement                                                            |
| ------------------ | ------------------------------------------------------------------------------- |
| Agent API          | Introduction of `create_agent` as standard way to build agents                  |
| Middleware         | Full customizable hooks around model/tool calls, prebuilt middleware available  |
| Structured Output  | Native support for typed output via schema + tool strategy                      |
| Content Blocks     | Unified `.content_blocks` across providers for reasoning, tool calls, citations |
| Package Namespace  | Lean top-level API; legacy moved to `langchain-classic`                         |
| Migration & Compat | Migration guide available; backward compatibility via legacy package            |
| Production Focus   | Built for agent engineering at scale, more stable, fewer breaking changes ahead |

---

## Implications for Your Bootcamp & Projects

Since you’re teaching an 8-week AI Engineering program and building agentic AI content, this v1.0 release aligns well because:

* You can focus course examples on the new `create_agent` paradigm rather than older chains/agents code.
* When teaching frameworks such as LangChain + LangGraph or RAG + agents, you can highlight how the new API supports middleware, structured output, and reasoning traces — important for production readiness.
* In your consulting services around LLMs, you can recommend clients adopt v1.0 for long-term stability, reducing custom legacy code and leveraging the unified abstractions.
* For participants already familiar with older LangChain versions (0.x), you can show migration paths and explain how the namespace changes and legacy modules map out.

---

[1]: https://docs.langchain.com/oss/python/releases/langchain-v1 "What's new in v1 - Docs by LangChain"
[2]: https://docs.langchain.com/oss/python/releases "Releases - Docs by LangChain"
[3]: https://docs.langchain.com/oss/python/migrate/langgraph-v1 "LangGraph v1 migration guide"
[4]: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/langchain-v1-is-now-generally-available/4462159 "LangChain v1 is now generally available!"
[5]: https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/ "LangChain & LangGraph 1.0 alpha releases"
