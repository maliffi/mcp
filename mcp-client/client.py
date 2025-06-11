import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session object
        self.session: Optional[ClientSession] = None
        
        # Initialize an AsyncExitStack to manage multiple async context managers
        # This helps ensure proper cleanup of async resources when they're no longer needed
        self.exit_stack = AsyncExitStack()
        
        # Initialize the Anthropic API client for interacting with Anthropic's AI services
        # The client will automatically use API credentials from environment variables
        #
        # This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        # - `api_key` from `ANTHROPIC_API_KEY`
        # - `auth_token` from `ANTHROPIC_AUTH_TOKEN`
        self.anthropic = Anthropic()
        
    # Connect to an MCP server
    # This method allows the client to communicate with the MCP server over standard input/output
    # It takes a path to the server script (.py or .js) as an argument
    # It uses the stdio_client to create a transport for the client to communicate with the server
    # It then creates a session for the client to interact with the server
    # It then lists the tools available on the server and prints them to the console
    # It then returns the tools
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # Create a new MCP client session using the stdio server parameters
        # This allows the client to communicate with the MCP server over standard input/output
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # Create a new session for the client to interact with the server
        # This allows the client to send and receive messages to/from the server
        self.stdio, self.write = stdio_transport
        client_session = ClientSession(self.stdio, self.write)
        # Assign the session just created to the mcp client
        self.session = await self.exit_stack.enter_async_context(client_session)

        # Initialize the session
        await self.session.initialize()

        # List available tools
        # This will return a list of tools that are available on the server
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to server with tools: {[tool.name for tool in tools]}")

        return tools
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Create a list of messages to send to the server
        # The first message is the user's query
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # List available tools
        # This will return a list of tools that are available on the server
        response = await self.session.list_tools()
        # Convert the tools to a list of dictionaries
        # This is the format that the Anthropic API expects
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        print(f"Available tools: {available_tools}")

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=os.getenv("ANTHROPIC_MODEL"),
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                print(f"Text content: {content.text}")
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                print(f"Tool use content: {content}")
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                print(f"Calling tool {tool_name} with args {tool_args}")
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"Tool result: {result}")
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL"),
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())