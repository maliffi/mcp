from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import argparse

# -- Constants
# Base URL for the NWS API
NWS_API_BASE = "https://api.weather.gov"
# User agent for requests to the NWS API
USER_AGENT = "weather-app/1.0"

# Initialize FastMCP server
mcp = FastMCP("weather")


# -- ---------------- --
# -- Helper functions --
# -- ---------------- --
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error making request to {url}", e)
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""
# -- ---------------- --

# -- ---------------- --
# -- MCP Tools --
# -- ---------------- --
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    # Call the NWS API to get the alerts for the state
    data = await make_nws_request(url)

    # Check if the request was successful
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    # Check if there are any alerts
    if not data["features"]:
        return "No active alerts for this state."

    # If there are alerts, then format them into a readable string
    alerts = [format_alert(feature) for feature in data["features"]]
    # Return the formatted alerts
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    # Check if the request was successful
    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    # Check if the request was successful
    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    # Only show next 5 periods
    for period in periods[:5]:
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    # Return the formatted forecast
    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    # Initialize and run the server
    # (transport='stdio' specifies that the server should use standard input/output streams for communication 
    # rather than other transport options like HTTP or WebSockets)
    
    

    # Production Mode
    # python weather.py --server_type=sse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )

    args = parser.parse_args()
    # -- Development Mode
    # mcp.run(transport='stdio')
    # -- Production Mode
    mcp.run(args.server_type)