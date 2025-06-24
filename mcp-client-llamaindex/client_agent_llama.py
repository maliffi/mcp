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

SYSTEM_PROMPT = """\
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with Our Database
"""

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
        description="An agent that can work with Our Database software.",
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