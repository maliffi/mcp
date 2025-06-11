# Weather Service

A Python-based weather service that provides real-time weather information using the National Weather Service (NWS) API. This service is built using FastMCP (Fast Model Control Protocol) and provides tools for fetching weather alerts and forecasts.

## What is MCP?

MCP (Model Control Protocol) is a protocol designed for building AI-powered services and tools. It provides a standardized way to:
- Create and expose tools that can be used by AI models
- Handle communication between AI models and services
- Manage the lifecycle of AI-powered tools and services

This weather service uses FastMCP, a Python implementation of MCP that makes it easy to create and expose weather-related tools to AI models.

## Features

- Get weather alerts for any US state
- Get detailed weather forecasts for specific locations using latitude and longitude coordinates
- Real-time data from the National Weather Service API
- Asynchronous API calls for better performance
- MCP-compatible tools that can be used by AI models
- Standard input/output stream communication for seamless integration

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- A virtual environment (recommended)
- MCP client or compatible AI model for interacting with the service

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd weather
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The service provides two MCP-compatible tools that can be used by AI models:

### 1. Get Weather Alerts
Get active weather alerts for any US state using its two-letter code.

Example:
```python
# Get alerts for California
alerts = await get_alerts("CA")
```

### 2. Get Weather Forecast
Get a detailed weather forecast for a specific location using latitude and longitude coordinates.

Example:
```python
# Get forecast for San Francisco (approximately)
forecast = await get_forecast(37.7749, -122.4194)
```

## Running the Service

To start the MCP server:

```bash
python weather.py
```

The service will start and listen for incoming requests using standard input/output streams, which is the default transport method for MCP communication.

## MCP Integration

This service is designed to be used as an MCP tool provider:
- Tools are decorated with `@mcp.tool()` to expose them to MCP clients
- Communication is handled through standard input/output streams
- All tools are properly documented with type hints and docstrings
- Error handling is implemented to ensure reliable communication

## API Details

The service uses the National Weather Service (NWS) API:
- Base URL: https://api.weather.gov
- All requests are made with proper error handling and timeouts
- Responses are formatted for easy reading
- All API calls are made asynchronously for better performance

## Dependencies

Main dependencies include:
- FastMCP: For building the MCP server and exposing tools
- httpx: For making asynchronous HTTP requests
- pydantic: For data validation
- Other supporting libraries (see requirements.txt for full list)

## Error Handling

The service includes built-in error handling for:
- API request failures
- Invalid coordinates
- Network timeouts
- Invalid state codes
- MCP communication errors

## Contributing

Feel free to submit issues and enhancement requests! When contributing, please ensure that:
- All new tools are properly decorated with `@mcp.tool()`
- Type hints and docstrings are included
- Error handling is implemented
- Tests are added for new functionality
