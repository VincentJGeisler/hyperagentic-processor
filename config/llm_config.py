"""
LLM Configuration for Hyperagentic Processor

This module configures the Language Model settings for all agents.
We use Groq as the LLM provider, just like in the original class example.
"""

import os
from typing import Dict, Any
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    # Look for .env file in project root (one level up from hyperagentic_processor)
    env_file = Path(__file__).parent.parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        # Also try in the hyperagentic_processor directory
        env_file_local = Path(__file__).parent.parent / ".env"
        if env_file_local.exists():
            with open(env_file_local, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            print(f"✅ Loaded environment variables from {env_file_local}")
        else:
            print(f"⚠️  No .env file found at {env_file} or {env_file_local}")

def get_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration for agents.
    
    Uses Groq API directly with current supported models.
    Updated to use llama-3.3-70b-versatile (current production model).
    """
    # Load .env file first
    load_env_file()
    
    # For AutoGen with Groq, we need to use a custom configuration
    return {
        "config_list": [
            {
                "model": "llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 4000,
        "timeout": 60
    }

def validate_llm_config() -> bool:
    """Validate that LLM configuration is properly set up"""
    config = get_llm_config()
    
    if not config["config_list"][0]["api_key"]:
        print("❌ GROQ_API_KEY not found")
        print("   Checked:")
        print("   1. Environment variable: GROQ_API_KEY")
        print("   2. .env file in project root")
        print("")
        print("   To fix:")
        print("   export GROQ_API_KEY='your_api_key_here'")
        print("   OR")
        print("   echo 'GROQ_API_KEY=your_api_key_here' > .env")
        return False
    
    print("✅ Groq LLM configuration validated")
    print(f"   Model: {config['config_list'][0]['model']}")
    print(f"   Base URL: {config['config_list'][0]['base_url']}")
    print(f"   API Key: {config['config_list'][0]['api_key'][:20]}...")
    return True

# Default configuration for testing (when API key not available)
TEST_LLM_CONFIG = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": "test_key_for_offline_testing",
            "base_url": "https://api.groq.com/openai/v1",
            "api_type": "openai"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 4000
}

if __name__ == "__main__":
    print("Hyperagentic Processor - LLM Configuration")
    print("=" * 50)
    validate_llm_config()