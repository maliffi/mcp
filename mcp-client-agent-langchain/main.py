# Create server parameters for stdio connection
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatOllama
import asyncio
import logging
from dotenv import load_dotenv
from config import Config
from mcp import ClientSession


# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

user_prompt = "What is the weather in NYC?"

def load_llm() -> ChatOllama:
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
    llm = ChatOllama(model=Config.LLM, request_timeout=Config.LLM_REQ_TIMEOUT_SECONDS)    
    logger.info(f"Loaded LLM: {Config.LLM}")
    return llm

async def run_agent(model: ChatOllama):
    async with sse_client(url=Config.MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({user_prompt})
            return agent_response

def main():
    model = load_llm() 
    result = asyncio.run(run_agent(model))
    print(result)


if __name__ == "__main__":
    main()
