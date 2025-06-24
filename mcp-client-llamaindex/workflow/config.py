"""
Configuration utility module that loads environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
ROOT_DIR = Path(__file__).parent

# Load environment variables from .env file
env_path = ROOT_DIR / '.env'
print(f"Loading environment variables from path: {env_path}")
load_dotenv(dotenv_path=env_path)

# Configuration class to access environment variables
class Config:
    """Configuration class that provides access to environment variables"""
    
    # The LLM model to use for generation
    LLM = os.getenv('LLM')
    
    # LLM settings
    LLM_REQ_TIMEOUT_SECONDS = float(os.getenv('LLM_REQ_TIMEOUT_SECONDS', '120.0'))

    # The MCP server URL
    MCP_SERVER_URL = os.getenv('MCP_SERVER_URL')
    
    # Application settings
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
