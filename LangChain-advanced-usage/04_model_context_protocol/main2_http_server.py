"""
HTTP MCP server with async weather tools.

Shows how to:
- Create an MCP server that runs as an HTTP service
- Define async tools that could call external APIs
- Use streamable-http transport for remote connectivity
- Handle multiple clients connecting to the same server
"""
import asyncio
import random
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Create an MCP server named "Weather"
# This name is exposed to MCP clients as the server identifier
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Get current weather for a given location.
    
    In a real implementation, this would call a weather API like OpenWeatherMap.
    For demo purposes, we return mock data.
    """
    # Simulate API call delay
    await asyncio.sleep(0.5)
    
    # Mock weather data
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
    temperatures = list(range(-10, 35))
    
    condition = random.choice(weather_conditions)
    temp = random.choice(temperatures)
    
    result = f"Weather in {location}: {condition.title()}, {temp}°C"
    print(f"[MCP Weather Server] {result}")
    return result

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """
    Get weather forecast for a given location and number of days.
    
    Args:
        location: The city or location to get forecast for
        days: Number of days to forecast (default: 3, max: 7)
    """
    # Validate days parameter
    days = min(max(days, 1), 7)  # Clamp between 1 and 7
    
    # Simulate API call delay
    await asyncio.sleep(1.0)
    
    forecast_data = []
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "thunderstorms"]
    
    for day in range(days):
        date_str = datetime.now().strftime(f"%Y-%m-%d +{day}d")
        condition = random.choice(weather_conditions)
        high_temp = random.randint(15, 30)
        low_temp = random.randint(5, high_temp - 5)
        
        forecast_data.append(f"Day {day + 1}: {condition.title()}, High: {high_temp}°C, Low: {low_temp}°C")
    
    result = f"Weather forecast for {location} ({days} days):\n" + "\n".join(forecast_data)
    print(f"[MCP Weather Server] Generated {days}-day forecast for {location}")
    return result

@mcp.tool()
async def get_weather_alerts(location: str) -> str:
    """
    Get weather alerts and warnings for a given location.
    """
    # Simulate API call delay
    await asyncio.sleep(0.3)
    
    # Mock alerts (in real implementation, this would come from weather service)
    alert_types = [
        "No active alerts",
        "Heavy rain warning until 6 PM",
        "High wind advisory in effect",
        "Heat wave warning for next 3 days",
        "Winter storm watch issued"
    ]
    
    alert = random.choice(alert_types)
    result = f"Weather alerts for {location}: {alert}"
    print(f"[MCP Weather Server] {result}")
    return result

if __name__ == "__main__":
    print("[MCP Weather Server] Starting HTTP weather server on localhost:8000...")
    print("[MCP Weather Server] MCP endpoint will be available at http://localhost:8000/mcp")
    
    # Run the server using streamable-http transport
    # This means:
    # - Runs a standalone HTTP server (default port 8000)
    # - Clients connect via HTTP POST/GET requests
    # - Supports multiple concurrent clients
    # - Supports streaming responses for large results
    # - Perfect for production and remote access
    mcp.run(transport="streamable-http")