## Context Engineering in Agents
## Overview
The hard part of building agents (or any LLM application) is making them reliable enough. While they may work for a prototype, they often fail in real-world use cases.
​
### Why do agents fail?
When agents fail, it’s usually because the LLM call inside the agent took the wrong action / didn’t do what we expected. LLMs fail for one of two reasons:
The underlying LLM is not capable enough
The “right” context was not passed to the LLM
More often than not - it’s actually the second reason that causes agents to not be reliable.

- **Context engineering is providing the right information and tools in the right format so the LLM can accomplish a task.** 

- **This is the number one job of AI Engineers.** This lack of “right” context is the number one blocker for more reliable agents, and LangChain’s agent abstractions are uniquely designed to facilitate context engineering.

## The agent loop
A typical agent loop consists of two main steps:
Model call - calls the LLM with a prompt and available tools, returns either a response or a request to execute tools
Tool execution - executes the tools that the LLM requested, returns tool results

## What you can control
- To build reliable agents, you need to control what happens at each step of the agent loop, as well as what happens between steps.

| Context Type        | What You Control                                                                 | Transient or Persistent |
|---------------------|-----------------------------------------------------------------------------------|--------------------------|
| **Model Context**   | What goes into model calls (instructions, message history, tools, response format) | Transient                |
| **Tool Context**    | What tools can access and produce (reads/writes to state, store, runtime context)  | Persistent               |
| **Life-cycle Context** | What happens between model and tool calls (summarization, guardrails, logging, etc.) | Persistent           |

### Transient context
- What the LLM sees for a single call. You can modify messages, tools, or prompts without changing what’s saved in state.

### Persistent context
- What gets saved in state across turns. Life-cycle hooks and tool writes modify this permanently.

## Data sources
Throughout this process, your agent accesses (reads / writes) different sources of data:

| Data Source        | Also Known As        | Scope                  | Examples                                                                     |
|--------------------|-----------------------|-------------------------|------------------------------------------------------------------------------|
| **Runtime Context** | Static configuration | Conversation-scoped     | User ID, API keys, database connections, permissions, environment settings   |
| **State**           | Short-term memory    | Conversation-scoped     | Current messages, uploaded files, authentication status, tool results        |
| **Store**           | Long-term memory     | Cross-conversation      | User preferences, extracted insights, memories, historical data              |

## How it works
LangChain middleware is the mechanism under the hood that makes context engineering practical for developers using LangChain.
Middleware allows you to hook into any step in the agent lifecycle and:
Update context
Jump to a different step in the agent lifecycle
Throughout this guide, you’ll see frequent use of the middleware API as a means to the context engineering end.

