## Structured output

- Structured output allows agents to return data in a specific, predictable format. Instead of parsing natural language responses, you get structured data in the form of JSON objects, Pydantic models, or dataclasses that your application can directly use.
- LangChain’s create_agent handles structured output automatically. The user sets their desired structured output schema, and when the model generates the structured data, it’s captured, validated, and returned in the 'structured_response' key of the agent’s state.
```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
    ]
```

## Response Format
- Controls how the agent returns structured data:
    - ToolStrategy[StructuredResponseT]: Uses tool calling for structured output
    - ProviderStrategy[StructuredResponseT]: Uses provider-native structured output
    - type[StructuredResponseT]: Schema type - automatically selects best strategy based on model capabilities
    - None: No structured output

- When a schema type is provided directly, LangChain automatically chooses:
    - ProviderStrategy for models supporting native structured output (e.g. OpenAI, Grok)
    - ToolStrategy for all other models

The structured response is returned in the structured_response key of the agent’s final state.

## Provider strategy
- Some model providers support structured output natively through their APIs (e.g. OpenAI, Grok, Gemini). This is the most reliable method when available.
To use this strategy, configure a ProviderStrategy:
```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
```

- Provider-native structured output provides high reliability and strict validation because the model provider enforces the schema. Use it when available.

```python
from pydantic import BaseModel, Field
# BaseModel: used to define typed output schemas.
# Field: lets you add descriptions and metadata for each field.

from langchain.agents import create_agent
# create_agent: builds a LangChain v1 agent with tools and structured output support.


# -----------------------------------------------------------
# DEFINE A STRUCTURED OUTPUT SCHEMA
# -----------------------------------------------------------
class ContactInfo(BaseModel):
    """Contact information for a person."""
    
    # Name field with description metadata
    name: str = Field(description="The name of the person")

    # Email field
    email: str = Field(description="The email address of the person")

    # Phone number field
    phone: str = Field(description="The phone number of the person")


# -----------------------------------------------------------
# CREATE THE AGENT WITH STRUCTURED OUTPUT
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5",       # The LLM capable of structured output
    tools=tools,         # Any tools you want to expose (not required)
    
    # response_format does the magic:
    # - Automatically switches to ProviderStrategy
    # - Forces the LLM to return a ContactInfo object
    response_format=ContactInfo  
)


# -----------------------------------------------------------
# INVOKE THE AGENT
# The model extracts structured data from a natural language input.
# -----------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"
    }]
})


# -----------------------------------------------------------
# READ STRUCTURED OUTPUT
# LangChain parses the LLM output into a ContactInfo object.
# -----------------------------------------------------------
result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

## Tool calling strategy
- For models that don’t support native structured output, LangChain uses tool calling to achieve the same result. This works with all models that support tool calling, which is most modern models.

```python
from typing import Generic, Union, Callable, TypeVar

SchemaT = TypeVar("SchemaT")


class ToolStrategy(Generic[SchemaT]):
    """
    A strategy object that defines **how a tool should behave inside an agent**.
    
    This includes:
    - what schema the tool should produce
    - what content should show up in the ToolMessage
    - how errors raised by the tool should be handled
    """

    # -----------------------------------------------------------
    # schema
    # -----------------------------------------------------------
    # The output schema (usually a Pydantic model or TypedDict)
    # that the tool is expected to conform to.
    #
    # Example:
    #   schema=ContactInfo  → tool must return ContactInfo structure.
    #
    # The agent uses this to:
    # - validate tool output
    # - guide the model to return structured data
    schema: type[SchemaT]

    # -----------------------------------------------------------
    # tool_message_content
    # -----------------------------------------------------------
    # Optional string representing what should be placed inside
    # the ToolMessage content when the tool is executed.
    #
    # If None → the default behavior is used.
    #
    # Example:
    #   tool_message_content="User info lookup completed"
    #
    # This content appears in:
    #   result["messages"][-1].content
    tool_message_content: str | None

    # -----------------------------------------------------------
    # handle_errors
    # -----------------------------------------------------------
    # Controls how tool exceptions should be h
```
### schema (required)
- The schema defining the structured output format. Supports:
    - Pydantic models: BaseModel subclasses with field validation
    - Dataclasses: Python dataclasses with type annotations
    - TypedDict: Typed dictionary classes
    - JSON Schema: Dictionary with JSON schema specification
    - Union types: Multiple schema options. The model will choose the most appropriate schema based on the context.

### tool_message_content
- Custom content for the tool message returned when structured output is generated. If not provided, defaults to a message showing the structured response data.

### handle_errors
Error handling strategy for structured output validation failures. Defaults to True.
    - True: Catch all errors with default error template
    - str: Catch all errors with this custom message
    - type[Exception]: Only catch this exception type with default message
    - tuple[type[Exception], ...]: Only catch these exception types with default message
    - Callable[[Exception], str]: Custom function that returns error message
    - False: No retry, let exceptions propagate

```python
from pydantic import BaseModel, Field
# BaseModel + Field → define strict structured schemas for LLM output.

from typing import Literal
# Literal → restricts a field to specific allowed string values.

from langchain.agents import create_agent
# create_agent → builds a LangChain v1 agent.

from langchain.agents.structured_output import ToolStrategy
# ToolStrategy(schema) → tells the agent to return structured output
# using the given schema and a tool-like parsing strategy.


# -----------------------------------------------------------
# DEFINE THE STRUCTURED OUTPUT MODEL
# -----------------------------------------------------------
class ProductReview(BaseModel):
    """Analysis of a product review."""

    # Product rating from 1 to 5.
    rating: int | None = Field(
        description="The rating of the product",
        ge=1,   # minimum value = 1
        le=5    # maximum value = 5
    )

    # Sentiment, restricted to two possible values.
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )

    # Short key points extracted from the review.
    # Model must return lowercase phrases, each 1–3 words.
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )


# -----------------------------------------------------------
# CREATE AN AGENT WITH STRUCTURED OUTPUT
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5",       # LLM used to generate the structured response
    tools=tools,         # Tools available to the agent (not required for structured output)

    # ToolStrategy(schema) → 
    # - forces LLM to output valid ProductReview data
    # - uses provider-native structured output mode
    # - validates and parses the final output
    response_format=ToolStrategy(ProductReview)
)


# -----------------------------------------------------------
# INVOKE THE AGENT
# Ask the model to analyze a natural-language review.
# -----------------------------------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"
    }]
})


# -----------------------------------------------------------
# ACCESS THE PARSED MODEL OUTPUT
# Returned as a proper ProductReview object.
# -----------------------------------------------------------
result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

## Custom tool message content
The tool_message_content parameter allows you to customize the message that appears in the conversation history when structured output is generated:
```python
from pydantic import BaseModel, Field
# BaseModel + Field → define strict structured outputs.

from typing import Literal
# Literal → restricts a field to allowed string values.

from langchain.agents import create_agent
# create_agent → builds a LangChain v1 agent.

from langchain.agents.structured_output import ToolStrategy
# ToolStrategy → forces the agent to output structured data,
# and optionally customizes how the tool-like output is recorded.


# -----------------------------------------------------------
# STRUCTURED OUTPUT MODEL
# Defines exactly what fields the LLM must return.
# -----------------------------------------------------------
class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""

    # Task text extracted from the meeting notes
    task: str = Field(description="The specific task to be completed")

    # Person responsible for completing the task
    assignee: str = Field(description="Person responsible for the task")

    # Priority level extracted from the transcript
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# -----------------------------------------------------------
# CREATE THE AGENT WITH STRUCTURED OUTPUT
# -----------------------------------------------------------
agent = create_agent(
    model="gpt-5",   # LLM used to generate meeting action items
    tools=[],        # No tools needed here

    # ToolStrategy:
    # - Forces the LLM to produce a MeetingAction object
    # - Adds a custom message into chat history when structured output is returned
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)


# -----------------------------------------------------------
# INVOKE THE AGENT
# Ask it to extract an action item from a natural-language sentence.
# The result will include:
#   - A MeetingAction structured object
#   - A ToolMessage saying: "Action item captured and added to meeting notes!"
# -----------------------------------------------------------
agent.invoke({
    "messages": [{
        "role": "user",
        "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"
    }]
})

```