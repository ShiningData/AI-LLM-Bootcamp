# Model Context
Control what goes into each model call - instructions, available tools, which model to use, and output format. These decisions directly impact reliability and cost.
- 1. System Prompt
- 2. Messages
- 3. Tools
- 4. Model
- 5. Response Format

## 1. System Prompt
- The system prompt sets the LLM’s behavior and capabilities. Different users, contexts, or conversation stages need different instructions. Successful agents draw on memories, preferences, and configuration to provide the right instructions for the current state of the conversation.

### 1.1. State-Aware Dynamic Prompt: 
- Access message count or conversation context from state:
```python
from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent.

from langchain.agents.middleware import dynamic_prompt, ModelRequest
# dynamic_prompt → dynamically generates the system prompt per request.
# ModelRequest   → contains request.state, request.messages, runtime, context, etc.


# -----------------------------------------------------------
# DYNAMIC PROMPT BASED ON CONVERSATION LENGTH
# -----------------------------------------------------------
@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """
    Generate a system prompt that adapts based on how long the
    conversation has been so far.

    request.messages → shortcut for request.state["messages"]
    """

    # Count how many messages are currently in the agent state
    message_count = len(request.messages)

    # Start with a baseline system prompt
    base = "You are a helpful assistant."

    # If the conversation is long, ask the LLM to be concise
    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base


# -----------------------------------------------------------
# BUILD THE AGENT WITH THE STATE-AWARE PROMPT
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                 # The main LLM
    tools=[...],                    # Tools the agent can call
    middleware=[state_aware_prompt] # Add dynamic system prompt middleware
)
```

### 1.2. Store
- Access user preferences from long-term memory:
```python
from dataclasses import dataclass
# dataclass → defines a simple structured context object

from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent

from langchain.agents.middleware import dynamic_prompt, ModelRequest
# dynamic_prompt → dynamically builds the system prompt per request
# ModelRequest   → includes runtime, state, context, messages, store, etc.

from langgraph.store.memory import InMemoryStore
# InMemoryStore → a key-value store used by LangGraph agents
#                 Useful for preferences, user memory, long-term state.


# -----------------------------------------------------------
# CUSTOM CONTEXT PASSED PER REQUEST
# -----------------------------------------------------------
@dataclass
class Context:
    user_id: str   # Identifier used to retrieve user-specific data from the store


# -----------------------------------------------------------
# STORE-AWARE DYNAMIC SYSTEM PROMPT
# -----------------------------------------------------------
@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """
    Build a dynamic system prompt based on user preferences
    retrieved from the agent's long-term storage.
    """

    # Read the user_id from the context
    user_id = request.runtime.context.user_id

    # Access the runtime store (InMemoryStore by default)
    store = request.runtime.store

    # Retrieve user preferences stored under namespace ("preferences",)
    user_prefs = store.get(("preferences",), user_id)
    # Keys in LangGraph store are tuples → ("preferences",), user_id

    # Base system prompt
    base = "You are a helpful assistant."

    # If preferences exist, customize the assistant behavior
    if user_prefs:
        # Example: {"communication_style": "concise"} or similar
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base


# -----------------------------------------------------------
# BUILD THE AGENT WITH STORE-AWARE PROMPTING
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                   # Main LLM
    tools=[...],                      # Tools the agent can call
    middleware=[store_aware_prompt],  # Dynamic system prompt using shared store
    context_schema=Context,           # Enables passing user_id per request
    store=InMemoryStore(),            # Persistent key-value store for preferences
)
```

### 1.3. Run-time Context:
- Access user ID or configuration from Runtime Context:
```python
from dataclasses import dataclass
# dataclass → simple container for structured context data

from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent

from langchain.agents.middleware import dynamic_prompt, ModelRequest
# dynamic_prompt → dynamically generates system prompt per request
# ModelRequest   → exposes runtime, context, messages, state, etc.


# -----------------------------------------------------------
# CUSTOM PER-REQUEST CONTEXT
# Passed to the agent to adapt its behavior
# -----------------------------------------------------------
@dataclass
class Context:
    user_role: str         # e.g., "admin", "viewer"
    deployment_env: str    # e.g., "production", "staging"


# -----------------------------------------------------------
# DYNAMIC PROMPT THAT ADAPTS TO USER ROLE + ENVIRONMENT
# -----------------------------------------------------------
@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """
    Build the system prompt by reading metadata from runtime.context.
    This allows the agent to behave differently for different users
    and different deployment environments.
    """

    # Read from Context passed at invocation time
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    # Base system instruction
    base = "You are a helpful assistant."

    # -------------------------------------------------------
    # Adjust behavior based on user role
    # -------------------------------------------------------
    if user_role == "admin":
        # Admins can do everything
        base += "\nYou have admin access. You can perform all operations."

    elif user_role == "viewer":
        # Viewers have restricted permissions
        base += "\nYou have read-only access. Guide users to read operations only."

    # -------------------------------------------------------
    # Adjust behavior based on deployment environment
    # -------------------------------------------------------
    if env == "production":
        # Adds caution when running in production systems
        base += "\nBe extra careful with any data modifications."

    return base


# -----------------------------------------------------------
# BUILD THE AGENT WITH CONTEXT-AWARE PROMPTING
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                     # Main LLM
    tools=[...],                        # Tools the agent can call
    middleware=[context_aware_prompt],  # Dynamic system prompt middleware
    context_schema=Context              # Required to pass structured context
)
```

## 2. Messages
- Messages make up the prompt that is sent to the LLM. It’s critical to manage the content of messages to ensure that the LLM has the right information to respond well.
- Transient vs Persistent Message Updates:
The examples above use wrap_model_call to make transient updates - modifying what messages are sent to the model for a single call without changing what’s saved in state.
For persistent updates that modify state (like the summarization example in Life-cycle Context), use life-cycle hooks like before_model or after_model to permanently update the conversation history. See the middleware documentation for more details.

### 2.1. State
- Inject uploaded file context from State when relevant to current query:
```python
from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent around a LangGraph workflow.

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# wrap_model_call → middleware that wraps the LLM call itself.
#                   Lets you edit messages, insert context, or override behavior.
# ModelRequest  → includes state, messages, runtime, context, etc.
# ModelResponse → the final LLM response object.

from typing import Callable
# Callable → used for typing the inner handler function.


# -----------------------------------------------------------
# WRAP THE MODEL CALL TO INJECT FILE CONTEXT
# -----------------------------------------------------------
@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Inject information about uploaded files into the LLM request.
    
    - Reads file metadata from the agent's state
    - Appends a user-visible message describing available files
    - The updated ModelRequest is passed to the underlying model call
    """

    # Pull "uploaded_files" list from the agent state
    # request.state is equivalent to AgentState
    uploaded_files = request.state.get("uploaded_files", [])

    # If user has uploaded files, add a contextual message
    if uploaded_files:

        # Build a bullet-point list summarizing each file
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        # Construct the final context block injected into the prompt
        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        # -------------------------------------------------------
        # Inject context as a final "user" message
        # -------------------------------------------------------
        # request.messages → the original sequence of messages
        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]

        # Modify the request with new messages using request.override
        request = request.override(messages=messages)

    # Finally, pass the modified request to the actual model
    return handler(request)


# -----------------------------------------------------------
# BUILD AGENT WITH FILE-AWARE MODEL MIDDLEWARE
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                  # Main LLM
    tools=[...],                     # Optional tools
    middleware=[inject_file_context] # Inject file metadata into model call
)
```

### 2.2. Store
- Inject user’s email writing style from Store to guide drafting:

```python
from dataclasses import dataclass
# dataclass → defines simple structured context passed into the agent

from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent (LangGraph under the hood)

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# wrap_model_call → middleware that wraps the actual LLM invocation.
# Allows modifying messages, injecting dynamic context, or overwriting behavior.
# ModelRequest  → contains messages, state, context, store, runtime
# ModelResponse → result returned by the underlying model

from typing import Callable
# Callable → for typing the handler function

from langgraph.store.memory import InMemoryStore
# InMemoryStore → persistent key/value store used by agent runtime
# Great for user memory and personalization.


# -----------------------------------------------------------
# CUSTOM CONTEXT PASSED ON EACH INVOCATION
# -----------------------------------------------------------
@dataclass
class Context:
    user_id: str   # Used to fetch user-specific writing styles from the store


# -----------------------------------------------------------
# WRAP THE MODEL CALL TO INJECT USER'S WRITING STYLE
# -----------------------------------------------------------
@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Inject the user's writing style (tone, greetings, examples) into the LLM request.
    The writing style is stored in the LangGraph Store and retrieved via user_id.
    """

    # Get the user ID from contextual metadata
    user_id = request.runtime.context.user_id

    # Access persistent store (InMemoryStore)
    store = request.runtime.store

    # Retrieve writing style under namespace ("writing_style",)
    writing_style = store.get(("writing_style",), user_id)

    # If the user has a stored style, inject it
    if writing_style:
        style = writing_style.value  # a dict like {"tone": "...", "greeting": "...", ...}

        # Build an instruction block describing the user's distinct style
        style_context = f"""Your writing style:
- Tone: {style.get('tone', 'professional')}
- Typical greeting: "{style.get('greeting', 'Hi')}"
- Typical sign-off: "{style.get('sign_off', 'Best')}"
- Example email you've written:
{style.get('example_email', '')}"""

        # IMPORTANT:
        # Append style context at the END → LLMs pay strong attention to trailing messages
        messages = [
            *request.messages,                             # retain original messages
            {"role": "user", "content": style_context}     # inject style preferences
        ]

        # Override the request with the augmented message list
        request = request.override(messages=messages)

    # Finally, delegate to the underlying actual model call
    return handler(request)


# -----------------------------------------------------------
# BUILD THE AGENT WITH WRITING STYLE PERSONALIZATION
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                     # Main LLM handling the conversation
    tools=[...],                        # Optional tools the agent can call
    middleware=[inject_writing_style],  # Middleware injecting style instructions
    context_schema=Context,             # Allows passing user_id into runtime.context
    store=InMemoryStore()               # Persistent user memory store
)
```

---

#### 🧠 **Personalization via Long-Term Memory**

You store user preferences (tone, signature, example emails) in the agent’s persistent store.

- Example store entry:

```python
store.set(("writing_style",), "user_123", {
    "tone": "casual",
    "greeting": "Hey team,",
    "sign_off": "Cheers",
    "example_email": "Hey team, just checking in on the status..."
})
```

#### ✉️ **Adaptive Writing Style**

The agent now writes emails:

* with the user's preferred tone
* using their greeting and signature
* following their writing patterns

This creates a **personalized AI writing assistant**.

#### ⚙️ **wrap_model_call = full control**

This middleware:

* intercepts the exact message sequence
* injects personalized context
* calls the model with the augmented request

Perfect for:

* email copilots
* CRM assistants
* onboarding assistants
* customer agent impersonation (within safe limits)

#### 🔁 **Context + Store = Personalized AI**

You combine:

* `context_schema` → identifies the user
* `store` → retrieves user-specific preferences
* dynamic context injection → influences each model call

This is enterprise-quality personalization.


### 2.3. Runtime Context
- Inject compliance rules from Runtime Context based on user’s jurisdiction:

```python
from dataclasses import dataclass
# dataclass → simple structured container for passing context into the agent

from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# wrap_model_call → wraps the LLM call itself, letting us modify messages
# ModelRequest   → includes messages, context, state, runtime, store
# ModelResponse  → the final response object returned by the model

from typing import Callable
# Callable → used for typing the underlying handler function


# -----------------------------------------------------------
# CUSTOM CONTEXT FOR COMPLIANCE-AWARE AGENTS
# -----------------------------------------------------------
@dataclass
class Context:
    user_jurisdiction: str                # e.g., "EU", "US", "UK"
    industry: str                         # e.g., "finance", "healthcare"
    compliance_frameworks: list[str]      # e.g., ["GDPR", "HIPAA"]


# -----------------------------------------------------------
# WRAP THE MODEL CALL TO INJECT COMPLIANCE RULES
# -----------------------------------------------------------
@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Inject compliance constraints into the LLM request,
    using regulatory metadata provided via Runtime Context.
    """

    # -------------------------------------------------------
    # Read compliance context metadata from runtime.context
    # -------------------------------------------------------
    jurisdiction = request.runtime.context.user_jurisdiction
    industry = request.runtime.context.industry
    frameworks = request.runtime.context.compliance_frameworks

    # -------------------------------------------------------
    # Build list of compliance rules based on frameworks + industry
    # -------------------------------------------------------
    rules = []

    # GDPR Rules
    if "GDPR" in frameworks:
        rules.append("- Must obtain explicit consent before processing personal data")
        rules.append("- Users have the right to request data deletion")

    # HIPAA Rules
    if "HIPAA" in frameworks:
        rules.append("- Cannot share patient health information without authorization")
        rules.append("- Must use secure, encrypted communication channels")

    # Industry-specific rules
    if industry == "finance":
        rules.append("- Cannot provide financial advice without proper disclaimers")

    # -------------------------------------------------------
    # Inject rules into the prompt if any are applicable
    # -------------------------------------------------------
    if rules:
        compliance_context = f"""Compliance requirements for {jurisdiction}:
{chr(10).join(rules)}"""

        # Append context at the END → LLM pays highest attention to trailing messages
        messages = [
            *request.messages,                             # original conversation
            {"role": "user", "content": compliance_context}  # injected compliance block
        ]

        # Override the request with updated message sequence
        request = request.override(messages=messages)

    # Pass modified request down to the underlying LLM call
    return handler(request)


# -----------------------------------------------------------
# BUILD AGENT WITH COMPLIANCE-INJECTION MIDDLEWARE
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                      # Main LLM
    tools=[...],                         # Optional tools
    middleware=[inject_compliance_rules],# Add compliance injection
    context_schema=Context               # Allow passing structured compliance context
)
```

---

#### 🧠 **Regulatory-aware LLM behavior**

The agent now auto-adjusts output based on:

* user jurisdiction
* industry domain
* compliance frameworks (GDPR, HIPAA, etc.)

#### ⚙️ **Dynamic prompt injection**

`wrap_model_call` allows you to:

* modify messages before LLM sees them
* inject compliance constraints
* prevent accidental violations

#### 🔐 **Enterprise use-case ready**

This pattern is ideal for building AI agents for:

* finance
* healthcare
* government services
* legal workflows
* customer support with compliance filters

#### 🧱 **No changes needed in agent logic**

All compliance behavior is handled *outside* the model and tools, purely via middleware.


## 3. Tools
- Tools let the model interact with databases, APIs, and external systems. How you define and select tools directly impacts whether the model can complete tasks effectively.

### Defining tools
- Each tool needs a clear name, description, argument names, and argument descriptions. These aren’t just metadata—they guide the model’s reasoning about when and how to use the tool.


### Selecting tools
Not every tool is appropriate for every situation. Too many tools may overwhelm the model (overload context) and increase errors; too few limit capabilities. Dynamic tool selection adapts the available toolset based on authentication state, user permissions, feature flags, or conversation stage.

### 3.1. State
- State-Based Tool Filtering Using `wrap_model_call`

```python
from langchain.agents import create_agent
# create_agent → builds the LangChain v1 agent (backed by LangGraph)

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# wrap_model_call → intercepts and modifies the LLM call before it executes
# ModelRequest   → contains messages, state, runtime, tools, context
# ModelResponse  → the output returned by the underlying model

from typing import Callable
# Callable → for defining the type of the inner handler function


# -----------------------------------------------------------
# MIDDLEWARE: FILTER TOOLS BASED ON STATE
# -----------------------------------------------------------
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Filter available tools dynamically based on conversation state.
    
    Examples:
    - If the user is not authenticated → only allow public tools.
    - If early in the conversation → disable advanced tools.
    """

    # -------------------------------------------------------
    # Access the agent's conversation state
    # -------------------------------------------------------
    state = request.state

    # Flag stored in state that indicates whether the user authenticated
    is_authenticated = state.get("authenticated", False)

    # Number of messages exchanged so far in this agent run
    message_count = len(state["messages"])


    # -------------------------------------------------------
    # TOOL FILTERING RULES
    # -------------------------------------------------------

    # Rule 1: If user is NOT authenticated → expose ONLY public tools
    if not is_authenticated:
        tools = [
            t for t in request.tools
            if t.name.startswith("public_")            # keep: public_search
        ]
        request = request.override(tools=tools)

    # Rule 2: If authenticated BUT conversation is still short → restrict advanced tools
    elif message_count < 5:
        tools = [
            t for t in request.tools
            if t.name != "advanced_search"             # hide advanced_search temporarily
        ]
        request = request.override(tools=tools)


    # -------------------------------------------------------
    # Pass modified request to the underlying model
    # -------------------------------------------------------
    return handler(request)


# -----------------------------------------------------------
# BUILD AGENT WITH STATE-BASED TOOL FILTERING
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-4o",                 # Main LLM
    tools=[public_search, private_search, advanced_search],  # All tools
    middleware=[state_based_tools], # Dynamic tool filtering middleware
)
```

---

#### 🔐 **Role-Based + State-Based Tool Access**

Tools are enabled/disabled depending on:

* authentication status
* conversation progress (message count)
* any state variables you choose

This is the foundation of **RBAC (Role-Based Access Control)** inside an LLM agent.

#### 🧠 **Dynamically changing the toolset**

`wrap_model_call` allows:

* modifying available tools
* injecting or removing functionality
* preventing misuse of sensitive tools

All without modifying the core agent.

#### 🚦 Practical enterprise controls

Examples supported by this pattern:

* Only authenticated users can use banking tools
* Advanced operations enabled after safety verification
* Specific tools disabled in production
* Tools limited in early conversation to reduce hallucinations

#### ⚙️ **Flexible and general**

`request.override(tools=...)` makes tool filtering extremely easy.


### 3.2. Store

...

### 3.3. Runtime context

...

## 4. Model
- Different models have different strengths, costs, and context windows. Select the right model for the task at hand, which might change during an agent run.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Initialize models once outside the middleware
large_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
efficient_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)  

    if message_count > 20:
        # Long conversation - use model with larger context window
        model = large_model
    elif message_count > 10:
        # Medium conversation
        model = standard_model
    else:
        # Short conversation - use efficient model
        model = efficient_model

    request = request.override(model=model)  

    return handler(request)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[...],
    middleware=[state_based_model]
)
```

## 5. Response Format
- Structured output transforms unstructured text into validated, structured data. When extracting specific fields or returning data for downstream systems, free-form text isn’t sufficient.
How it works: When you provide a schema as the response format, the model’s final response is guaranteed to conform to that schema. The agent runs the model / tool calling loop until the model is done calling tools, then the final response is coerced into the provided format.
### 5.1. Defining formats
Schema definitions guide the model. Field names, types, and descriptions specify exactly what format the output should adhere to.
```python
from pydantic import BaseModel, Field

class CustomerSupportTicket(BaseModel):
    """Structured ticket information extracted from customer message."""

    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency level: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(
        description="One-sentence summary of the customer's issue"
    )
    customer_sentiment: str = Field(
        description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'"
    )
```

### 5.2. Selecting Formats
- Dynamic response format selection adapts schemas based on user preferences, conversation stage, or role—returning simple formats early and detailed formats as complexity increases.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    """Simple response for early conversation."""
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    """Detailed response for established conversation."""
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on State."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)  

    if message_count < 3:
        # Early conversation - use simple format
        request = request.override(response_format=SimpleResponse)  
    else:
        # Established conversation - use detailed format
        request = request.override(response_format=DetailedResponse)  

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_based_output]
)
```