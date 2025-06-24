import nest_asyncio
import asyncio


from mcp.client.sse import sse_client
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from mcp import ClientSession
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import logging
from mcp.types import Tool
from config import Config
from contextlib import AsyncExitStack

from llama_index.core.llms import ChatMessage

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_llm():
    """
    Load the LLM from the configuration and set it as the default LLM in settings.
    Args:
        None
    Returns:
        Ollama: The loaded LLM instance.
    """
    # Specifying a request_timeout of Config.LLM_REQ_TIMEOUT_SECONDS seconds 
    # for requests to the LLM to ensure that the system doesn't get stuck 
    # if the model takes too long to respond.
    logger.info(f"Loading LLM: {Config.LLM}, request_timeout: {Config.LLM_REQ_TIMEOUT_SECONDS}")
    llm = Ollama(model=Config.LLM, request_timeout=Config.LLM_REQ_TIMEOUT_SECONDS)    
    logger.info(f"Loaded LLM: {Config.LLM}")
    return llm

class MCPClient:
    def __init__(self, mcp_server_url: str):
        # Load the LLM
        self.llm = load_llm()
        self.mcp_server_url = mcp_server_url

        # Initialize an AsyncExitStack to manage multiple async context managers
        # This helps ensure proper cleanup of async resources when they're no longer needed
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self) -> list[Tool]:
        """Connect to an MCP server"""
        # Create a new MCP client session using the sse server parameters
        # This allows the client to communicate with the MCP server over sse
        sse_transport = await self.exit_stack.enter_async_context(sse_client(self.mcp_server_url))

        # Create a new session for the client to interact with the server
        # This allows the client to send and receive messages to/from the server
        self.sse, self.write = sse_transport
        client_session = ClientSession(self.sse, self.write)
        # Assign the session just created to the mcp client
        self.session = await self.exit_stack.enter_async_context(client_session)

        # Initialize the session
        await self.session.initialize()

        # List available tools
        # This will return a list of tools that are available on the server
        tools_response = await self.session.list_tools()
        tools = tools_response.tools
        logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")

        # Convert each FunctionalTool into a Ollama Tool

        return tools

    
    # ----START------------------------------------------------------------
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Create a list of messages to send to the server
        # The first message is the user's query
        messages = [
            ChatMessage(
                role= "user",
                content= query
            )
        ]

        # List available tools
        # This will return a list of tools that are available on the server
        response = await self.session.list_tools()
        # Convert the tools to a list of dictionaries
        # This is the format that the LLM API expects
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        logger.info(f"Available tools: {available_tools}")

        # Initial LLM API call with tools
        response = self.llm.chat(messages, tools=available_tools)
        logger.info(f"LLM Response: {response}")

        # Process response and handle tool calls
        final_text = []
        
        # Handle the response message
        if hasattr(response, 'message'):
            message = response.message
            if hasattr(message, 'content'):
                final_text.append(message.content)
                logger.info(f"Text content: {message.content}")
                
                # Check if the response contains tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments
                        
                        # Execute tool call
                        logger.info(f"Calling tool {tool_name} with args {tool_args}")
                        result = await self.session.call_tool(tool_name, tool_args)
                        logger.info(f"Tool result: {result}")
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        
                        # Add tool result to messages
                        messages.append(ChatMessage(role="assistant", content=message.content))
                        messages.append(ChatMessage(role="user", content=f"Tool {tool_name} returned: {result.content}"))
                        
                        # Get next response from LLM with tools
                        response = self.llm.chat(messages, tools=available_tools)
                        if hasattr(response, 'message') and hasattr(response.message, 'content'):
                            final_text.append(response.message.content)

        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info("\nMCP Client Started!")
        logger.info("Type your queries or 'quit' to exit.")

        
        try:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                logger.info("Exiting chat loop.")
                return
            
            response = await self.process_query(query)
            logger.info("\n" + response)

        except Exception as e:
            logger.error(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
    # ----END------------------------------------------------------------


    async def handle_user_message(
        message_content: str,
        agent: FunctionAgent,
        agent_context: Context,
        verbose: bool = False,
    ):
        """Handle a user message using the agent."""
        handler = agent.run(message_content, ctx=agent_context)
        async for event in handler.stream_events():
            if verbose and type(event) == ToolCall:
                print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
            elif verbose and type(event) == ToolCallResult:
                print(f"Tool {event.tool_name} returned {event.tool_output}")

        response = await handler
        return str(response)

async def main():
    # Initialize MCP client and tool spec
    mcp_client = MCPClient(mcp_server_url=Config.MCP_SERVER_URL)
    
    try:
        await mcp_client.connect_to_server()
        logger.info(f"Connected to server {Config.MCP_SERVER_URL}")
        await mcp_client.chat_loop()
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 