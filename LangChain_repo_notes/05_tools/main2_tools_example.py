from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import json

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai", max_tokens=500)

# 1. Basic tool definition with @tool decorator
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    customers = ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Wilson"]
    matches = [c for c in customers if query.lower() in c.lower()][:limit]
    return f"Found {len(matches)} customers: {', '.join(matches)}"

# 2. Custom tool name and description
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        # Safe eval for basic math only
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            return f"{expression} = {eval(expression)}"
        else:
            return "Error: Only basic math operations allowed"
    except:
        return "Error: Invalid expression"

# 3. Advanced schema definition with Pydantic
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
    result = f"Current weather in {location}: {temp}° {units.title()}"
    
    if include_forecast:
        result += "\n5-day forecast: Sunny, Cloudy, Rainy, Sunny, Partly Cloudy"
    
    return result

def main():
    # Create agent with all our tools
    agent = create_agent(
        model=model,
        tools=[search_database, calc, get_weather]
    )
    
    print("🛠️ LangChain Tools Demo")
    print("Available tools:")
    print("1. search_database - Find customers")
    print("2. calculator - Do math")
    print("3. get_weather - Weather info")
    print("\nType 'quit' to exit\n")
    
    conversation = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        conversation.append({"role": "user", "content": user_input})
        
        try:
            result = agent.invoke({"messages": conversation})
            response = result["messages"][-1].content
            
            # Clean up response format
            if isinstance(response, list) and len(response) > 0:
                response = response[0].get('text', str(response))
                
            conversation.append({"role": "assistant", "content": response})
            print(f"Agent: {response}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()