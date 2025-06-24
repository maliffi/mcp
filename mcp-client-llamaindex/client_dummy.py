from llama_index.tools.mcp import BasicMCPClient
import os
from dotenv import load_dotenv
import logging
import asyncio
from llama_index.llms.ollama import Ollama

load_dotenv()


# Set up logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL')),
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
    logger.info(f"Loading LLM: {os.getenv('LLM')}, request_timeout: {os.getenv('LLM_REQ_TIMEOUT_SECONDS')}")
    llm = Ollama(model=os.getenv('LLM'), request_timeout=os.getenv('LLM_REQ_TIMEOUT_SECONDS'))    
    logger.info(f"Loaded LLM: {os.getenv('LLM')}")
    return llm

async def main():
    llm = load_llm()
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    # Connect to an MCP server using different transports
    http_client = BasicMCPClient(mcp_server_url)  # Server-Sent Events    

    # List available tools
    tools = await http_client.list_tools()
    logger.info(f"Tools: {tools}")

    # List available prompts
    prompts = await http_client.list_prompts()
    logger.info(f"Prompts: {prompts}")

    # Call a tool
    result = await http_client.call_tool("get_alerts", {"state": "NY"})
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())