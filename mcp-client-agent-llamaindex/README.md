# MCP Client Agent with LlamaIndex

This project implements a function-calling agent workflow using LlamaIndex, connected to the Model Control Protocol (MCP). It demonstrates how to create an interactive agent that can use tools via function calling to interact with MCP servers.

## Overview

The implementation is based on the LlamaIndex function calling agent workflow example:
https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agentworkflow/function_calling_agent/

## Features

- Utilizes LlamaIndex's workflow system for agent creation
- Implements function calling to allow the agent to use tools
- Connects to MCP servers via BasicMCPClient
- Supports chat memory and streaming responses
- Uses Ollama for local LLM inference

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   - Create a `.env` file with necessary configuration
   - Set the MCP_SERVER_URL and other required parameters

3. Ensure your LLM (Ollama) is properly set up and running

## Usage

Run the main agent workflow:
```bash
python workflow_function_calling_agent.py
```

The agent will:
1. Connect to the configured MCP server
2. Load the specified LLM model via Ollama
3. Prompt for user input
4. Process requests, potentially using MCP tools
5. Return responses with any relevant sources/citations

## Project Structure

- `workflow_function_calling_agent.py`: Main agent implementation
- `agent_events.py`: Event definitions for the workflow
- `config.py`: Configuration settings
- `requirements.txt`: Project dependencies
- `pyproject.toml`: Python project metadata

## License

Please refer to the original LlamaIndex examples repository for licensing information.
