## Create tools
### Basic tool definition
The simplest way to create a tool is with the @tool decorator. By default, the function’s docstring becomes the tool’s description that helps the model understand when to use it:
```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```
- Type hints are required as they define the tool’s input schema. **The docstring should be informative and concise to help the model understand the tool’s purpose.**

#### Customize tool properties
##### Custom tool name and Custom tool description
By default, the tool name comes from the function name. Override it when you need something more descriptive:
```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

### Advanced schema definition
- Define complex inputs with Pydantic models or JSON schemas:
```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```
### Example
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/05_tools/main2_tools_example.py 
🛠️ LangChain Tools Demo
Available tools:
1. search_database - Find customers
2. calculator - Do math
3. get_weather - Weather info

Type 'quit' to exit

You:
```

#### Sample questions:
 For search_database:
  - "Find customers named John"
  - "Search for customers with 'Smith' in their name"

  For calculator:
  - "What's 15 + 27?"
  - "Calculate 100 / 4"

  For get_weather:
  - "What's the weather in Paris?"
  - "Get weather for Tokyo in fahrenheit with forecast"

  Mix them:
  - "Search for Alice and calculate 5 * 8"
  - "Weather in London and find customer Bob"

  Start with something simple like: "What's 25 + 17?"