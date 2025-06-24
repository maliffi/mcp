from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import logging
from config import Config
from llama_index.core.base.llms.types import ChatMessage

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with National weather service.

When a user asks a question:
1. Determine if you need to use a tool to answer
2. If yes, execute immediatly the tool call with the correct parameters
3. After receiving the tool results, provide a natural language response to the user based on those results
4. Do NOT just return the raw tool output or tool calls
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

    
    def process_query(self, query: str, messages: list[ChatMessage] = []) -> tuple[str, list[ChatMessage]]:
        """Process user prompt and return response """
        # Add the user's query to the messages list
        messages.append(
            ChatMessage(
                role= "user",
                content= query
            )
        )
        response = self.llm.chat(messages, tools=self.tools)
        return response, messages
    

def main():
    # Initialize MCP client and tool spec
    mcp_client = MCPClient(mcp_server_url=Config.MCP_SERVER_URL)
    # Main interaction loop
    print("\nEnter 'exit' to quit")
    while True:
        try:
            user_input = input("\nEnter your message: ")
            if user_input.lower() == "exit":
                break
                
            print(f"\nUser: {user_input}")
            messages = [
                ChatMessage(
                    role="system",
                    content=SYSTEM_PROMPT
                )
            ]
            response, messages = mcp_client.process_query(user_input, messages)
            print(f"MCP: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 