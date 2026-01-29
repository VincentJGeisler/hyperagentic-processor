"""
MCP Configuration Management Module

Handles loading, validation, and management of MCP ecosystem configuration
from .kiro/settings/mcp.json and environment variables.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("MCPConfig")


class MCPConfigurationManager:
    """
    Manages Oracle MCP ecosystem configuration.
    
    Responsibilities:
    - Load configuration from .kiro/settings/mcp.json
    - Handle environment variables for API keys
    - Provide validation and defaults
    - Support dynamic configuration updates
    - Notify watchers of configuration changes
    """
    
    def __init__(self, config_path: str = ".kiro/settings/mcp.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file relative to project root
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.watchers: List[Callable] = []
        self.lock = asyncio.Lock()
        self._initialized = False
        
    async def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from file and environment variables.
        
        Returns:
            Loaded configuration dictionary
        """
        async with self.lock:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                self.config = self._get_default_configuration()
                await self.save_configuration()
            else:
                try:
                    with open(config_file, 'r') as f:
                        self.config = json.load(f)
                    logger.info(f"Configuration loaded from {self.config_path}")
                except Exception as e:
                    logger.error(f"Failed to load configuration: {e}")
                    self.config = self._get_default_configuration()
            
            # Load API keys from environment variables
            self._load_api_keys_from_env()
            
            # Validate configuration
            self._validate_configuration()
            
            self._initialized = True
            return self.config
    
    async def save_configuration(self):
        """Save configuration to file."""
        async with self.lock:
            config_file = Path(self.config_path)
            
            # Create directory if it doesn't exist
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Update last_updated timestamp
                self.config["last_updated"] = datetime.now().isoformat()
                
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                logger.info(f"Configuration saved to {self.config_path}")
                
                # Notify watchers
                for watcher in self.watchers:
                    try:
                        if asyncio.iscoroutinefunction(watcher):
                            await watcher(self.config)
                        else:
                            watcher(self.config)
                    except Exception as e:
                        logger.error(f"Configuration watcher failed: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                raise
    
    async def update_configuration(
        self,
        path: str,
        value: Any
    ) -> bool:
        """
        Update specific configuration value.
        
        Args:
            path: Dot-notation path (e.g., "web_search.default_provider")
            value: New value
            
        Returns:
            True if update successful, False otherwise
        """
        async with self.lock:
            try:
                keys = path.split('.')
                target = self.config
                
                # Navigate to parent of target key
                for key in keys[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                
                # Update the target key
                target[keys[-1]] = value
                
                logger.info(f"Configuration updated: {path} = {value}")
                
                # Save configuration
                await self.save_configuration()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to update configuration at {path}: {e}")
                return False
    
    def get_configuration(self, path: Optional[str] = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Dot-notation path (e.g., "web_search.providers.brave.enabled")
                  If None, returns entire configuration
                  
        Returns:
            Configuration value or None if not found
        """
        if not self._initialized:
            logger.warning("Configuration not initialized, returning empty dict")
            return {} if path is None else None
            
        if path is None:
            return self.config
            
        keys = path.split('.')
        target = self.config
        
        try:
            for key in keys:
                if isinstance(target, dict) and key in target:
                    target = target[key]
                else:
                    return None
            return target
        except Exception as e:
            logger.error(f"Error getting configuration at {path}: {e}")
            return None
    
    def add_watcher(self, callback: Callable):
        """
        Add configuration change watcher.
        
        Args:
            callback: Function to call when configuration changes.
                     Can be sync or async function.
        """
        self.watchers.append(callback)
        logger.debug(f"Added configuration watcher: {callback.__name__}")
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables."""
        providers = self.config.get("web_search", {}).get("providers", {})
        
        for provider_name, provider_config in providers.items():
            api_key_env = provider_config.get("api_key_env")
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if api_key:
                    # Store reference that key is available
                    provider_config["api_key_available"] = True
                    logger.info(f"API key loaded for {provider_name} from {api_key_env}")
                else:
                    provider_config["api_key_available"] = False
                    logger.warning(f"API key not found for {provider_name} in env var {api_key_env}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider from environment.
        
        Args:
            provider: Provider name (e.g., "brave", "tavily", "serpapi")
            
        Returns:
            API key string or None if not found
        """
        providers = self.config.get("web_search", {}).get("providers", {})
        provider_config = providers.get(provider, {})
        api_key_env = provider_config.get("api_key_env")
        
        if api_key_env:
            return os.getenv(api_key_env)
        return None
    
    def _validate_configuration(self):
        """Validate configuration structure and values."""
        required_sections = [
            "web_search",
            "security"
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing required configuration section: {section}")
                # Add default section
                if section == "web_search":
                    self.config[section] = self._get_default_web_search_config()
                elif section == "security":
                    self.config[section] = self._get_default_security_config()
        
        # Validate web search providers
        web_search = self.config.get("web_search", {})
        providers = web_search.get("providers", {})
        
        if not providers:
            logger.warning("No web search providers configured")
        
        # Validate cache settings
        cache_config = web_search.get("cache", {})
        if cache_config.get("enabled"):
            ttl = cache_config.get("ttl_seconds", 3600)
            if ttl < 0:
                logger.warning(f"Invalid cache TTL: {ttl}, setting to default 3600")
                cache_config["ttl_seconds"] = 3600
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "web_search": self._get_default_web_search_config(),
            "security": self._get_default_security_config(),
            "monitoring": {
                "enabled": True,
                "log_level": "INFO"
            }
        }
    
    def _get_default_web_search_config(self) -> Dict[str, Any]:
        """Get default web search configuration."""
        return {
            "enabled": True,
            "providers": {
                "brave": {
                    "enabled": True,
                    "priority": 1,
                    "api_key_env": "BRAVE_API_KEY",
                    "rate_limit": 2000,
                    "api_key_available": False
                },
                "tavily": {
                    "enabled": True,
                    "priority": 2,
                    "api_key_env": "TAVILY_API_KEY",
                    "rate_limit": 1000,
                    "api_key_available": False
                },
                "serpapi": {
                    "enabled": True,
                    "priority": 3,
                    "api_key_env": "SERPAPI_KEY",
                    "rate_limit": 100,
                    "api_key_available": False
                }
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 3600,
                "max_size_mb": 50
            },
            "security": {
                "allowed_domains": ["*"],
                "blocked_domains": [],
                "max_results": 10
            }
        }
    
    def _get_default_security_config(self) -> Dict[str, Any]:
        """Get default security configuration."""
        return {
            "sandbox_enabled": True,
            "network_gateway": {
                "enabled": True,
                "whitelist_only": False,
                "allowed_domains": [
                    "api.search.brave.com",
                    "api.tavily.com",
                    "serpapi.com"
                ],
                "require_https": True
            },
            "resource_limits": {
                "max_total_mcp_memory_mb": 300,
                "max_mcp_processes": 5,
                "max_open_files": 100
            }
        }


# Global configuration manager instance
_config_manager: Optional[MCPConfigurationManager] = None


def get_config_manager(config_path: str = ".kiro/settings/mcp.json") -> MCPConfigurationManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = MCPConfigurationManager(config_path)
    return _config_manager


async def initialize_config(config_path: str = ".kiro/settings/mcp.json") -> Dict[str, Any]:
    """
    Initialize configuration system.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = get_config_manager(config_path)
    return await manager.load_configuration()


if __name__ == "__main__":
    # Test configuration manager
    import asyncio
    
    async def test_config():
        print("MCP Configuration Manager Test")
        print("=" * 50)
        
        # Initialize configuration
        config = await initialize_config()
        
        print(f"\nConfiguration loaded:")
        print(f"  Version: {config.get('version')}")
        print(f"  Web Search Enabled: {config.get('web_search', {}).get('enabled')}")
        
        # Test provider configuration
        providers = config.get('web_search', {}).get('providers', {})
        print(f"\nConfigured providers: {len(providers)}")
        for name, provider in providers.items():
            print(f"  {name}:")
            print(f"    Priority: {provider.get('priority')}")
            print(f"    Enabled: {provider.get('enabled')}")
            print(f"    API Key Available: {provider.get('api_key_available')}")
        
        # Test getting specific configuration
        manager = get_config_manager()
        cache_ttl = manager.get_configuration("web_search.cache.ttl_seconds")
        print(f"\nCache TTL: {cache_ttl} seconds")
        
        # Test updating configuration
        success = await manager.update_configuration("web_search.cache.ttl_seconds", 7200)
        print(f"\nConfiguration update: {'Success' if success else 'Failed'}")
        
        new_ttl = manager.get_configuration("web_search.cache.ttl_seconds")
        print(f"New Cache TTL: {new_ttl} seconds")
    
    asyncio.run(test_config())
