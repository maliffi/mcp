import asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import logging
from config import Config
from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# What is the weather in NYC?

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
    # By assigning the Ollama model instance to Settings.llm, any component in the LlamaIndex ecosystem 
    # that needs to use an LLM will automatically use this instance unless explicitly overridden.
    Settings.llm = llm
    return llm

class MCPClient(BasicMCPClient):
    def __init__(self, mcp_server_url: str):
        # Load the LLM
        self.llm = load_llm()
        # Initialize the MCP client
        llama_index_mcp_client = BasicMCPClient(mcp_server_url)
        # Initialize the MCP tool spec
        mcp_tool_spec = McpToolSpec(client=llama_index_mcp_client)
        function_tools = mcp_tool_spec.to_tool_list()
        logger.info(f"Connected to server with tools: {[func_tool.metadata.name for func_tool in function_tools]}")
        self.function_tools = function_tools

        # Extract raw functions from FunctionTool objects
        tools = []
        for func_tool in function_tools:
            func = func_tool.fn
            if not hasattr(func, '__name__'):
                func.__name__ = func_tool.metadata.name
            if not hasattr(func, '__doc__'):
                func.__doc__ = func_tool.metadata.description
            tools.append(func)
        self.tools = tools

async def get_agent(tools: McpToolSpec, llm: Ollama):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with National Weather Service.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    # Create a handler for streaming the agent's steps
    handler = await agent.run(message_content, ctx=agent_context)
    
    # Track tools and their results for manual execution if needed
    tool_calls = []
    tool_results = []
    
    # Stream events to see the agent's progress
    async for event in handler.stream_events():
        logger.info(f"Event type: {type(event)}, event: {event}\n\n------\n\n")
        if type(event) == ToolCall:
            tool_calls.append(event)
            if verbose:
                print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif type(event) == ToolCallResult:
            tool_results.append(event)
            if verbose:
                print(f"Tool {event.tool_name} returned {event.tool_output}")
    
    # Get the final response after tool execution
    final_response = await handler
    logger.info(f"Final response: {final_response}")
    
    # If the response is a JSON object with tool_calls and no actual response,
    # we need to manually process the tool results and generate a final response
    if str(final_response).strip().startswith('{"tool_calls"'):
        if len(tool_results) > 0:
            # If we have tool results, we can use them to generate a response
            # Create a new prompt with the tool results
            tool_result_message = f"Tool results: \n"
            for result in tool_results:
                tool_result_message += f"{result.tool_name} returned: {result.tool_output}\n"
            
            # Ask the LLM to provide a natural language response based on the tool results
            final_prompt = f"""Based on the following tool results, provide a natural language response to the user's question: '{message_content}'
            
            {tool_result_message}
            
            Response to user:"""
            
            response = await Settings.llm.acomplete(final_prompt)
            return response.text
        else:
            # If we don't have tool results but we have tool calls,
            # we need to manually execute the tools
            return f"I found information that might help answer your question about: '{message_content}', but I need to process it further."
    
    # Return the final response
    return str(final_response)

    

async def main():
    # Initialize MCP client and tool spec
    llm = load_llm()
    mcp_client = BasicMCPClient(Config.MCP_SERVER_URL)
    mcp_tool = McpToolSpec(client=mcp_client)

    # get the agent
    agent = await get_agent(mcp_tool, llm)

    # create the agent context
    agent_context = Context(agent)

    # Run the agent!
    while True:
        user_input = input("Enter your message: ")
        if user_input == "exit":
            break
        print("User: ", user_input)
        response = await handle_user_message(user_input, agent, agent_context, verbose=True)
        print("Agent: ", response)

if __name__ == "__main__":
    asyncio.run(main())