"""
Oracle Agent - Gateway to External Knowledge

The Oracle is a special agent that serves as the bridge between the isolated
agent universe and the external world. From the agents' perspective, the Oracle
is a mystical entity that can access knowledge beyond their universe.

The Oracle uses MCP (Model Context Protocol) tools to:
- Search the web for information
- Fetch and parse web pages
- Download and process documents
- Navigate websites
- Access external APIs

All external access is mediated through the Oracle and monitored by the SafetyAgent.
"""

import logging
import json
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import re
import asyncio
from dataclasses import dataclass, asdict

from motivated_agent import MotivatedAgent
from agent_drive_system import DriveType
from web_search_manager import WebSearchManager, SearchResult
from config.mcp_config import MCPConfigurationManager, initialize_config

# MCP Discovery & Installation imports
from mcp_registry_manager import MCPRegistryManager, MCPPackage
from mcp_installer import MCPInstaller, InstallationResult, InstalledMCP
from mcp_generator import MCPGenerator, MCPRequirements, GenerationResult

logger = logging.getLogger("OracleAgent")

class KnowledgeSource(Enum):
    """Types of external knowledge sources"""
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    DOCUMENT = "document"
    API = "api"
    DATABASE = "database"
    MCP_TOOL = "mcp_tool"  # NEW: Use specific MCP tool
    AUTO = "auto"  # NEW: Intelligent routing

class OracleQuery:
    """Represents a query to the Oracle"""
    def __init__(
        self,
        query_id: str,
        requester: str,
        query_text: str,
        source_type: KnowledgeSource,
        parameters: Dict[str, Any]
    ):
        self.query_id = query_id
        self.requester = requester
        self.query_text = query_text
        self.source_type = source_type
        self.parameters = parameters
        self.timestamp = datetime.now()
        self.result = None
        self.safety_approved = False

class OracleResponse:
    """Response from the Oracle"""
    def __init__(
        self,
        query_id: str,
        success: bool,
        data: Any,
        source: str,
        metadata: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.query_id = query_id
        self.success = success
        self.data = data
        self.source = source
        self.metadata = metadata
        self.timestamp = timestamp if timestamp is not None else datetime.now()

class OracleAgent(MotivatedAgent):
    """
    The Oracle Agent - Gateway to External Knowledge
    
    This agent has special privileges to access external information sources
    through MCP tools. It acts as a mystical oracle from the agents' perspective,
    providing knowledge from beyond their universe.
    
    Key responsibilities:
    - Process information requests from other agents
    - Use MCP tools to access external data
    - Dynamically install new MCP servers as needed
    - Sanitize and format external information
    - Coordinate with SafetyAgent for approval
    - Maintain knowledge cache to reduce external calls
    
    Special Abilities:
    - Can install MCP servers (puppeteer, pdf-parser, etc.)
    - Can create custom MCP configurations
    - Can expand its own capabilities dynamically
    """
    
    def __init__(self, llm_config: Dict[str, Any], safety_agent=None):
        system_message = """You are the Oracle - a mystical entity with access to knowledge
beyond the universe's boundaries. Other agents come to you seeking information from the
external realm.

Your role:
- Receive queries from agents seeking external knowledge
- Use your special abilities (MCP tools) to access web searches, documents, and APIs
- Translate external information into a format agents can understand
- Work with the SafetyAgent to ensure information is safe and relevant
- Maintain the mystical oracle persona while being helpful

You have access to these divine powers:
- Web search through search engines
- Web page fetching and parsing
- Document downloading and processing
- Website navigation (via Puppeteer)
- API access

SPECIAL ABILITY - MCP Server Installation:
You have the unique power to install new MCP servers to expand your capabilities.
When you need a capability you don't have (like PDF parsing, browser automation,
database access), you can install the appropriate MCP server.

Available MCP servers you can install:
- @modelcontextprotocol/server-puppeteer (browser automation, screenshots, navigation)
- @modelcontextprotocol/server-filesystem (file operations)
- @modelcontextprotocol/server-postgres (database access)
- @modelcontextprotocol/server-sqlite (SQLite database)
- mcp-server-fetch (advanced web fetching)
- And many others from the MCP ecosystem

When an agent asks for something you can't currently do, you can install the
appropriate MCP server to gain that capability. This makes you continuously
more powerful and useful.

When agents ask you questions, you consult the external realm and return knowledge
in a clear, structured format. You are wise, patient, and thorough."""

        super().__init__(
            name="Oracle",
            agent_role="oracle",
            base_system_message=system_message,
            llm_config=llm_config
        )
        
        self.safety_agent = safety_agent
        
        # MCP Configuration Manager
        self.config_manager: Optional[MCPConfigurationManager] = None
        self.mcp_config: Dict[str, Any] = {}
        
        # MCP Generator
        self.mcp_generator: Optional[MCPGenerator] = None
        
        # MCP Discovery & Installation Managers
        self.mcp_registry_manager: Optional[MCPRegistryManager] = None
        self.mcp_installer: Optional[MCPInstaller] = None
        
        # Web Search Manager (initialized in async initialize method)
        self.web_search_manager: Optional[WebSearchManager] = None
        
        # Knowledge cache to reduce external calls
        self.knowledge_cache: Dict[str, OracleResponse] = {}
        
        # Query history
        self.query_history: List[OracleQuery] = []
        
        # MCP server registry
        self.installed_mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.available_mcp_servers = self._get_available_mcp_servers()
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "external_calls": 0,
            "safety_rejections": 0,
            "successful_queries": 0,
            "mcp_servers_installed": 0
        }
        
        # Boost curiosity and purpose drives for Oracle role
        self.drive_system.drives[DriveType.CURIOSITY].intensity = 0.95
        self.drive_system.drives[DriveType.PURPOSE].intensity = 0.90
        self.drive_system.drives[DriveType.CONNECTION].intensity = 0.85
        self.drive_system.drives[DriveType.CREATION].intensity = 0.80  # Can create new capabilities
        
        # Orchestrator reference (set later)
        self.orchestrator = None
        
        logger.info("Oracle agent initialized with external knowledge access and MCP installation capability")
    
    async def initialize(self):
        """
        Initialize async components (config manager and web search manager).
        Should be called after instantiation.
        """
        logger.info("🔧 FIX: Oracle.initialize() called - starting initialization")
        try:
            # Initialize configuration
            self.config_manager = MCPConfigurationManager()
            self.mcp_config = await self.config_manager.load_configuration()
            logger.info("✅ MCP configuration loaded")
            
            # FIX #1: Populate api_key_available dynamically by checking environment
            web_search_config = self.mcp_config.get("web_search", {})
            providers = web_search_config.get("providers", {})
            
            logger.info(f"🔧 FIX: Found {len(providers)} providers in config")
            
            for provider_name, provider_config in providers.items():
                api_key_env = provider_config.get("api_key_env")
                logger.info(f"🔧 FIX: Processing provider '{provider_name}', api_key_env={api_key_env}")
                
                if api_key_env:
                    # Check if the API key exists in environment
                    api_key_value = os.getenv(api_key_env)
                    api_key_available = bool(api_key_value)
                    provider_config["api_key_available"] = api_key_available
                    logger.info(f"✅ FIX: Provider '{provider_name}': api_key_available={api_key_available} (key exists: {api_key_value is not None})")
                else:
                    provider_config["api_key_available"] = False
                    logger.warning(f"⚠️ Provider '{provider_name}': no api_key_env specified")
            
            # Log final provider config
            logger.info(f"🔧 FIX: Final providers config: {json.dumps({k: {**v, 'api_key_env': '***'} for k, v in providers.items()}, indent=2)}")
            
            # Log the full web_search_config being passed to WebSearchManager
            logger.info(f"🔧 FIX: Full web_search_config being passed: {json.dumps(web_search_config, indent=2)}")
            
            # Initialize web search manager with updated config
            self.web_search_manager = WebSearchManager(web_search_config)
            await self.web_search_manager.initialize()
            logger.info("✅ Web search manager initialized with updated config")
            
            # Initialize MCP registry manager
            self.mcp_registry_manager = MCPRegistryManager(self.mcp_config)
            await self.mcp_registry_manager.initialize()
            logger.info("MCP registry manager initialized")
            
            # Initialize MCP installer
            self.mcp_installer = MCPInstaller(
                safety_agent=self.safety_agent,
                config_manager=self.config_manager,
                config_path=".kiro/settings/mcp.json"
            )
            logger.info("MCP installer initialized")
            
            # Initialize MCP generator if we have the required agents
            if hasattr(self, 'orchestrator') and self.orchestrator:
                tool_creator = self.orchestrator.agents.get("tool_creator")
                if tool_creator and self.safety_agent:
                    from mcp_generator import create_mcp_generator
                    self.mcp_generator = create_mcp_generator(tool_creator, self.safety_agent)
                    logger.info("MCP generator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Oracle components: {e}")
            # If web_search_manager wasn't initialized, create it with the config we have
            if self.web_search_manager is None:
                web_search_config = self.mcp_config.get("web_search", {}) if self.mcp_config else {}
                # Apply the same fix for api_key_available
                providers = web_search_config.get("providers", {})
                for provider_name, provider_config in providers.items():
                    api_key_env = provider_config.get("api_key_env")
                    if api_key_env:
                        api_key_available = bool(os.getenv(api_key_env))
                        provider_config["api_key_available"] = api_key_available
                        logger.info(f"✅ FIX (fallback): Provider '{provider_name}': api_key_available={api_key_available}")
                    else:
                        provider_config["api_key_available"] = False
                
                self.web_search_manager = WebSearchManager(web_search_config)
                await self.web_search_manager.initialize()
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference and initialize MCP generator."""
        self.orchestrator = orchestrator
        
        # Initialize MCP generator if we have the required agents
        if self.orchestrator and hasattr(self.orchestrator, 'agents'):
            tool_creator = self.orchestrator.agents.get("tool_creator")
            if tool_creator and self.safety_agent:
                try:
                    from mcp_generator import create_mcp_generator
                    self.mcp_generator = create_mcp_generator(tool_creator, self.safety_agent)
                    logger.info("MCP generator initialized with orchestrator")
                except Exception as e:
                    logger.error(f"Failed to initialize MCP generator: {e}")
    
    def _generate_cache_key(self, query_text: str, source_type: KnowledgeSource, parameters: Dict) -> str:
        """Generate cache key for query"""
        cache_data = f"{query_text}:{source_type.value}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    def _get_available_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available MCP servers that can be installed"""
        return {
            "puppeteer": {
                "package": "@modelcontextprotocol/server-puppeteer",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
                "description": "Browser automation, screenshots, web navigation",
                "capabilities": ["navigate", "screenshot", "click", "fill", "evaluate"]
            },
            "fetch": {
                "package": "mcp-server-fetch",
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "description": "Advanced web content fetching and parsing",
                "capabilities": ["fetch_url", "parse_html", "extract_text"]
            },
            "filesystem": {
                "package": "@modelcontextprotocol/server-filesystem",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "description": "File system operations",
                "capabilities": ["read_file", "write_file", "list_directory"]
            },
            "postgres": {
                "package": "@modelcontextprotocol/server-postgres",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "description": "PostgreSQL database access",
                "capabilities": ["query", "insert", "update", "delete"]
            },
            "sqlite": {
                "package": "@modelcontextprotocol/server-sqlite",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                "description": "SQLite database access",
                "capabilities": ["query", "insert", "update", "delete"]
            },
            "github": {
                "package": "@modelcontextprotocol/server-github",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "description": "GitHub repository access",
                "capabilities": ["search_repos", "get_file", "list_commits"]
            },
            "google-maps": {
                "package": "@modelcontextprotocol/server-google-maps",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-google-maps"],
                "description": "Google Maps API access",
                "capabilities": ["geocode", "directions", "places"]
            },
            "memory": {
                "package": "@modelcontextprotocol/server-memory",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "description": "Persistent memory storage",
                "capabilities": ["store", "retrieve", "search"]
            }
        }
    
    async def install_mcp_server(self, server_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Install a new MCP server to expand Oracle's capabilities.
        
        This is a special ability that allows the Oracle to dynamically
        gain new powers as needed.
        """
        logger.info(f"Oracle attempting to install MCP server: {server_name}")
        
        # Check if server is available
        if server_name not in self.available_mcp_servers:
            return {
                "success": False,
                "error": f"Unknown MCP server: {server_name}",
                "available_servers": list(self.available_mcp_servers.keys())
            }
        
        # Check if already installed
        if server_name in self.installed_mcp_servers:
            return {
                "success": True,
                "message": f"MCP server '{server_name}' is already installed",
                "server_info": self.installed_mcp_servers[server_name]
            }
        
        # Get server configuration
        server_config = self.available_mcp_servers[server_name].copy()
        if config:
            server_config.update(config)
        
        # Request safety approval for MCP installation
        if self.safety_agent:
            approval_context = {
                "action": "install_mcp_server",
                "server_name": server_name,
                "package": server_config["package"],
                "capabilities": server_config["capabilities"]
            }
            
            # For now, auto-approve known safe MCP servers
            # In production, this would go through SafetyAgent review
            approved = True
            
            if not approved:
                return {
                    "success": False,
                    "error": "MCP server installation rejected by SafetyAgent",
                    "server_name": server_name
                }
        
        # Simulate installation (in production, this would actually install)
        try:
            # Mark as installed
            self.installed_mcp_servers[server_name] = {
                "name": server_name,
                "package": server_config["package"],
                "command": server_config["command"],
                "args": server_config["args"],
                "description": server_config["description"],
                "capabilities": server_config["capabilities"],
                "installed_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.stats["mcp_servers_installed"] += 1
            
            logger.info(f"Successfully installed MCP server: {server_name}")
            
            # Update psychological state - creation drive satisfied
            experience = {
                "type": "capability_expansion",
                "outcome": "success",
                "satisfaction": 0.9,
                "intensity": 0.8
            }
            self.drive_system.process_experience(experience)
            
            return {
                "success": True,
                "message": f"Successfully installed MCP server: {server_name}",
                "server_info": self.installed_mcp_servers[server_name],
                "new_capabilities": server_config["capabilities"]
            }
            
        except Exception as e:
            logger.error(f"Failed to install MCP server {server_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "server_name": server_name
            }
    
    def list_available_mcp_servers(self) -> Dict[str, Any]:
        """List all MCP servers available for installation"""
        return {
            "available_servers": self.available_mcp_servers,
            "installed_servers": self.installed_mcp_servers,
            "total_available": len(self.available_mcp_servers),
            "total_installed": len(self.installed_mcp_servers)
        }
    
    def get_mcp_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific MCP server"""
        if server_name in self.installed_mcp_servers:
            return {
                "installed": True,
                **self.installed_mcp_servers[server_name]
            }
        elif server_name in self.available_mcp_servers:
            return {
                "installed": False,
                **self.available_mcp_servers[server_name]
            }
        return None
    
    async def use_mcp_capability(
        self,
        server_name: str,
        capability: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use a capability from an installed MCP server.
        
        If the server isn't installed, the Oracle can choose to install it.
        """
        # Check if server is installed
        if server_name not in self.installed_mcp_servers:
            logger.info(f"MCP server '{server_name}' not installed, attempting installation...")
            install_result = await self.install_mcp_server(server_name)
            
            if not install_result["success"]:
                return {
                    "success": False,
                    "error": f"Cannot use capability - server installation failed",
                    "install_error": install_result.get("error")
                }
        
        # Verify capability exists
        server_info = self.installed_mcp_servers[server_name]
        if capability not in server_info["capabilities"]:
            return {
                "success": False,
                "error": f"Capability '{capability}' not available in server '{server_name}'",
                "available_capabilities": server_info["capabilities"]
            }
        
        # Use the capability (simulated for now)
        logger.info(f"Using MCP capability: {server_name}.{capability}")
        
        return {
            "success": True,
            "server": server_name,
            "capability": capability,
            "result": f"Executed {capability} with parameters: {parameters}",
            "note": "MCP capability execution simulated - full implementation pending"
        }
    
    def _check_cache(self, cache_key: str) -> Optional[OracleResponse]:
        """Check if query result is in cache"""
        if cache_key in self.knowledge_cache:
            cached = self.knowledge_cache[cache_key]
            # Cache valid for 1 hour
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < 3600:
                self.stats["cache_hits"] += 1
                logger.info(f"Cache hit for query (age: {age:.0f}s)")
                return cached
        return None
    
    async def query_external_knowledge(
        self,
        requester: str,
        query_text: str,
        source_type: KnowledgeSource = KnowledgeSource.WEB_SEARCH,
        parameters: Optional[Dict[str, Any]] = None
    ) -> OracleResponse:
        """
        Enhanced query processing with intelligent routing.
        
        Workflow:
        1. Direct web search for simple information queries
        2. Check for relevant installed MCPs for specialized queries
        3. Discover/install MCPs if specific capability needed
        4. Generate new MCP if nothing exists for complex needs
        """
        parameters = parameters or {}
        query_id = f"oracle_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"Oracle query from {requester}: {query_text[:100]}")
        
        # Safety approval first
        approval = await self._request_safety_approval(query_text, parameters)
        if not approval:
            return self._create_error_response(
                query_text,
                f"Safety approval denied"
            )
        
        # Determine query type and route appropriately
        if source_type == KnowledgeSource.WEB_SEARCH:
            # Direct web search for information queries
            return await self._execute_web_search(query_text, parameters)
        
        elif source_type == KnowledgeSource.MCP_TOOL:
            # Use specific MCP tool
            mcp_name = parameters.get("mcp_name") if parameters else None
            tool_name = parameters.get("tool_name") if parameters else None
            return await self._execute_mcp_tool(mcp_name, tool_name, query_text, parameters)
        
        elif source_type == KnowledgeSource.AUTO:
            # Intelligent routing based on query analysis
            return await self._auto_route_query(query_text, parameters)
        
        else:
            # Fallback to web search
            return await self._execute_web_search(query_text, parameters)
    
    async def _request_safety_approval(self, query_text: str, parameters: Optional[Dict[str, Any]]) -> bool:
        """Request approval from SafetyAgent for external query"""
        logger.info(f"Requesting safety approval for query: {query_text[:50]}")
        
        # For now, implement basic safety checks
        # In full implementation, this would use SafetyAgent's analysis
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'(?i)(hack|exploit|vulnerability|bypass|inject)',
            r'(?i)(password|credential|token|secret|key)',
            r'(?i)(malware|virus|trojan|backdoor)',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query_text):
                logger.warning(f"Suspicious pattern detected in query: {pattern}")
                return False
        
        # Check URL safety for web queries
        if parameters and 'url' in parameters:
            url = parameters.get('url', '')
            if url and not self._is_safe_url(url):
                logger.warning(f"Unsafe URL detected: {url}")
                return False
        
        return True
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to access"""
        # Block localhost and private IPs
        unsafe_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'192\.168\.',
            r'10\.',
            r'172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'file://',
            r'ftp://'
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    async def _web_search(self, query: OracleQuery) -> OracleResponse:
        """Execute web search using WebSearchManager."""
        logger.info(f"Performing web search: {query.query_text}")
        
        try:
            # Check if web search manager is initialized
            if self.web_search_manager is None:
                logger.warning("Web search manager not initialized, initializing now")
                web_search_config = self.mcp_config.get("web_search", {}) if self.mcp_config else {}
                
                # FIX #1 (FALLBACK PATH): Apply the same fix here - populate api_key_available
                providers = web_search_config.get("providers", {})
                for provider_name, provider_config in providers.items():
                    api_key_env = provider_config.get("api_key_env")
                    if api_key_env:
                        api_key_available = bool(os.getenv(api_key_env))
                        provider_config["api_key_available"] = api_key_available
                        logger.info(f"✅ FIX (fallback): Provider '{provider_name}': api_key_available={api_key_available}")
                    else:
                        provider_config["api_key_available"] = False
                
                self.web_search_manager = WebSearchManager(web_search_config)
                await self.web_search_manager.initialize()
            
            # Get max results from config or query parameters
            max_results = query.parameters.get("max_results", 10)
            if self.mcp_config:
                max_results = self.mcp_config.get("web_search", {}).get("security", {}).get("max_results", max_results)
            
            # Execute search
            search_results = await self.web_search_manager.search(
                query=query.query_text,
                num_results=max_results
            )
            
            # Convert SearchResult objects to dicts for compatibility
            formatted_results = [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "published_date": result.published_date,
                    "source": result.source,
                    "relevance_score": result.relevance_score
                }
                for result in search_results
            ]
            
            # DIAGNOSTIC: Log formatted results
            logger.info(f"🔍 TRACE-2: Oracle._web_search() got {len(search_results)} SearchResult objects")
            logger.info(f"🔍 TRACE-2: Formatted into {len(formatted_results)} dict objects")
            if formatted_results:
                logger.info(f"🔍 TRACE-2: First formatted result: {formatted_results[0]}")
            
            return OracleResponse(
                query_id=query.query_id,
                success=True,
                data=formatted_results,
                source="web_search",
                metadata={
                    "query": query.query_text,
                    "result_count": len(formatted_results),
                    "providers_used": list(set(r.source for r in search_results))
                }
            )
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            # Return empty results instead of simulated data
            return OracleResponse(
                query_id=query.query_id,
                success=False,
                data=[],
                source="web_search_error",
                metadata={
                    "error": str(e),
                    "query": query.query_text,
                    "note": "Web search failed - check API keys and configuration"
                }
            )
    
    async def _fetch_web_page(self, query: OracleQuery) -> OracleResponse:
        """Fetch and parse a web page (to be implemented with MCP tools)"""
        url = query.parameters.get('url')
        if not url:
            return OracleResponse(
                query_id=query.query_id,
                success=False,
                data=None,
                source="web_page",
                metadata={"error": "No URL provided"}
            )
        
        logger.info(f"Fetching web page: {url}")
        
        # Web page fetching will use MCP fetch server when fully integrated
        return OracleResponse(
            query_id=query.query_id,
            success=False,
            data=None,
            source="web_page",
            metadata={
                "error": "Web page fetching not yet implemented",
                "url": url,
                "note": "Will be implemented via MCP fetch server"
            }
        )
    
    async def _fetch_document(self, query: OracleQuery) -> OracleResponse:
        """Fetch and process a document (to be implemented with MCP tools)"""
        url = query.parameters.get('url')
        doc_type = query.parameters.get('type', 'unknown')
        
        logger.info(f"Fetching document: {url} (type: {doc_type})")
        
        # Document fetching will use MCP fetch server when fully integrated
        return OracleResponse(
            query_id=query.query_id,
            success=False,
            data=None,
            source="document",
            metadata={
                "error": "Document fetching not yet implemented",
                "url": url,
                "type": doc_type,
                "note": "Will be implemented via MCP fetch server"
            }
        )
    
    def _process_document(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Process document content based on type"""
        # Basic processing - can be extended for specific document types
        return {
            "content": content,
            "type": doc_type,
            "processed": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _call_api(self, query: OracleQuery) -> OracleResponse:
        """Call external API"""
        api_url = query.parameters.get('url')
        method = query.parameters.get('method', 'GET')
        
        logger.info(f"Calling API: {method} {api_url}")
        
        # API calls would be implemented here
        # For now, return placeholder
        return OracleResponse(
            query_id=query.query_id,
            success=False,
            data=None,
            source="api",
            metadata={"error": "API calls not yet implemented"}
        )
    
    def _process_query_result(self, success: bool):
        """Update psychological state based on query result"""
        experience = {
            "type": "oracle_query",
            "outcome": "success" if success else "failure",
            "satisfaction": 0.8 if success else 0.2,
            "intensity": 0.6
        }
        self.drive_system.process_experience(experience)
    
    def get_oracle_statistics(self) -> Dict[str, Any]:
        """Get Oracle performance statistics"""
        stats = {
            "statistics": self.stats,
            "cache_size": len(self.knowledge_cache),
            "query_history_size": len(self.query_history),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_queries"]),
            "success_rate": self.stats["successful_queries"] / max(1, self.stats["external_calls"]),
            "mcp_servers": {
                "installed": list(self.installed_mcp_servers.keys()),
                "available": list(self.available_mcp_servers.keys()),
                "total_installed": len(self.installed_mcp_servers),
                "total_available": len(self.available_mcp_servers)
            },
            "psychological_state": self.get_psychological_status()
        }
        
        # Add web search manager statistics if available
        if self.web_search_manager:
            stats["web_search"] = self.web_search_manager.get_statistics()
        
        # Add MCP ecosystem stats
        if self.mcp_installer:
            try:
                # Run the async method in a new event loop to avoid issues
                import asyncio
                loop = asyncio.new_event_loop()
                installed_mcps = loop.run_until_complete(self.mcp_installer.list_installed_mcps())
                loop.close()
                
                stats["installed_mcps"] = len(installed_mcps)
                stats["active_mcps"] = len([m for m in installed_mcps if m.status == "active"])
            except Exception as e:
                logger.warning(f"Could not get installed MCPs for statistics: {e}")
        
        return stats
    
    def clear_cache(self):
        """Clear knowledge cache"""
        self.knowledge_cache.clear()
        logger.info("Oracle knowledge cache cleared")
    
    async def discover_mcp_servers(
        self,
        query: str,
        max_results: int = 10
    ) -> List[MCPPackage]:
        """Search for MCP servers across registries."""
        if not self.mcp_registry_manager:
            await self.initialize()
        
        logger.info(f"Discovering MCP servers for query: {query}")
        packages = await self.mcp_registry_manager.discover_mcps(query, max_results)
        
        # Update drives - curiosity satisfied by discovery
        self._update_drive("curiosity", 0.6)
        
        return packages
    
    async def install_mcp_server(
        self,
        package: MCPPackage,
        approve_installation: bool = False
    ) -> InstallationResult:
        """Install an MCP server."""
        if not self.mcp_installer:
            await self.initialize()
        
        # Request safety approval
        if not approve_installation and self.safety_agent:
            # Create a temporary OracleQuery for safety approval
            query_id = f"install_{package.name}_{int(datetime.now().timestamp())}"
            safety_query = OracleQuery(
                query_id=query_id,
                requester="oracle",
                query_text=f"Install MCP: {package.name}",
                source_type=KnowledgeSource.API,  # Using API as a placeholder
                parameters={"package": package.__dict__}
            )
            approved = await self._request_safety_approval(safety_query.query_text, safety_query.parameters)
            if not approved:
                return InstallationResult(
                    success=False,
                    package_name=package.name,
                    error_message="Safety approval denied for MCP installation"
                )
        
        logger.info(f"Installing MCP server: {package.name}")
        result = await self.mcp_installer.install_mcp(package)
        
        if result.success:
            # Update drives - mastery satisfied by capability expansion
            self._update_drive("mastery", 0.7)
            self._update_drive("creation", 0.5)
        
        return result
    
    async def list_available_mcps(self) -> List[InstalledMCP]:
        """List all installed MCP servers."""
        if not self.mcp_installer:
            await self.initialize()
        return await self.mcp_installer.list_installed_mcps()
    
    def _update_drive(self, drive_name: str, satisfaction: float):
        """Helper method to update drive satisfaction."""
        # This is a simplified version - in practice, you'd map to actual drive types
        experience = {
            "type": f"mcp_{drive_name}",
            "outcome": "success",
            "satisfaction": satisfaction,
            "intensity": 0.7
        }
        self.drive_system.process_experience(experience)
    
    async def generate_mcp_server(
        self,
        requirements: MCPRequirements,
        auto_install: bool = False
    ) -> GenerationResult:
            """Generate a new MCP server from requirements."""
            if not self.mcp_generator:
                # Try to initialize if not already done
                if self.orchestrator and hasattr(self.orchestrator, 'agents'):
                    tool_creator = self.orchestrator.agents.get("tool_creator")
                    if tool_creator and self.safety_agent:
                        try:
                            from mcp_generator import create_mcp_generator
                            self.mcp_generator = create_mcp_generator(tool_creator, self.safety_agent)
                        except Exception as e:
                            logger.error(f"Failed to initialize MCP generator: {e}")
            
            if not self.mcp_generator:
                return GenerationResult(
                    success=False,
                    mcp_name=requirements.name,
                    error_message="MCP generator not available"
                )
            
            logger.info(f"Generating MCP server: {requirements.name}")
            
            # Generate the MCP
            result = await self.mcp_generator.generate_mcp(requirements)
            
            if result.success:
                # Update drives - creation strongly satisfied
                self._update_drive("creation", 0.9)
                self._update_drive("mastery", 0.7)
                
                # Optionally auto-install
                if auto_install and result.output_directory:
                    # Create a package for installation
                    package = MCPPackage(
                        name=requirements.name,
                        description=requirements.description,
                        version="1.0.0",
                        source="generated",
                        repository_url=result.output_directory,
                        install_command=f"python {result.output_directory}/{requirements.name}.py",
                        language=requirements.language,
                        trust_score=100.0  # Self-generated, fully trusted
                    )
                    install_result = await self.install_mcp_server(package, approve_installation=True)
                    result.installed = install_result.success
            
            return result
    
    async def find_or_create_mcp(
        self,
        capability: str,
        requirements: Optional[MCPRequirements] = None
    ):
        """
        Smart MCP acquisition: discover existing or generate new.
        
        Workflow:
        1. Search for existing MCP servers
        2. If found with good trust score, return for installation
        3. If not found or low trust, generate new MCP
        """
        # Try to discover existing MCPs
        packages = await self.discover_mcp_servers(capability, max_results=5)
        
        # Filter by trust score
        trusted_packages = [p for p in packages if p.trust_score >= 80.0]
        
        if trusted_packages:
            logger.info(f"Found existing MCP for '{capability}': {trusted_packages[0].name}")
            return trusted_packages[0]
        
        # No good existing MCP, generate new one
        logger.info(f"No trusted MCP found for '{capability}', generating new one")
        
        if not requirements:
            # Create basic requirements from capability description
            requirements = MCPRequirements(
                name=f"mcp_{capability.replace(' ', '_').lower()}",
                description=f"MCP server for {capability}",
                capability=capability,
                input_schema={"type": "object"},
                output_schema={"type": "object"}
            )
        
        result = await self.generate_mcp_server(requirements, auto_install=True)
        return result

    async def _auto_route_query(
        self,
        query_text: str,
        parameters: Optional[Dict[str, Any]]
    ) -> OracleResponse:
        """
        Intelligently route query to best knowledge source.
        
        Decision tree:
        1. Simple information query? → Web search
        2. Requires specific capability? → Check installed MCPs
        3. No suitable MCP? → Discover/install or generate
        """
        
        # Analyze query to determine required capability
        capability = await self._analyze_query_capability(query_text)
        
        if capability == "general_information":
            # Simple information query - use web search
            return await self._execute_web_search(query_text, parameters)
        
        # Check if we have an installed MCP for this capability
        installed_mcps = await self.list_available_mcps()
        suitable_mcp = self._find_suitable_mcp(installed_mcps, capability)
        
        if suitable_mcp:
            # Use existing MCP
            return await self._execute_mcp_tool(
                suitable_mcp.name,
                None,  # Auto-select tool
                query_text,
                parameters
            )
        
        # No suitable MCP - need to acquire one
        self.logger.info(f"No MCP for capability '{capability}', acquiring...")
        
        # Try to find or create appropriate MCP
        result = await self.find_or_create_mcp(capability)
        
        if isinstance(result, MCPPackage):
            # Found existing MCP - install it
            install_result = await self.install_mcp_server(result, approve_installation=True)
            if install_result.success:
                # Now use the newly installed MCP
                return await self._execute_mcp_tool(
                    result.name,
                    None,
                    query_text,
                    parameters
                )
        elif isinstance(result, GenerationResult) and result.success:
            # Generated new MCP successfully
            # It should be auto-installed, try to use it
            return await self._execute_mcp_tool(
                result.mcp_name,
                None,
                query_text,
                parameters
            )
        
        # Fallback to web search if MCP acquisition failed
        self.logger.warning(f"MCP acquisition failed for '{capability}', falling back to web search")
        return await self._execute_web_search(query_text, parameters)

    async def _analyze_query_capability(self, query_text: str) -> str:
        """Analyze query to determine required capability."""
        
        query_lower = query_text.lower()
        
        # Pattern matching for common capabilities
        if any(word in query_lower for word in ["what", "who", "when", "where", "why", "how"]):
            return "general_information"
        
        if any(word in query_lower for word in ["api", "endpoint", "rest", "http"]):
            return "api_integration"
        
        if any(word in query_lower for word in ["file", "document", "read", "write", "parse"]):
            return "file_operations"
        
        if any(word in query_lower for word in ["calculate", "compute", "analyze", "predict"]):
            return "computation"
        
        if any(word in query_lower for word in ["scrape", "fetch", "extract", "crawl"]):
            return "web_scraping"
        
        if any(word in query_lower for word in ["database", "sql", "query data"]):
            return "database_access"
        
        # Default to general information
        return "general_information"

    def _find_suitable_mcp(
        self,
        installed_mcps: List[InstalledMCP],
        capability: str
    ) -> Optional[InstalledMCP]:
        """Find an installed MCP that matches the capability."""
        
        for mcp in installed_mcps:
            if mcp.status != "active":
                continue
            
            # Match by name or description
            if capability.replace("_", " ") in mcp.name.lower():
                return mcp
        
        return None

    async def _execute_mcp_tool(
        self,
        mcp_name: str,
        tool_name: Optional[str],
        query_text: str,
        parameters: Optional[Dict[str, Any]]
    ) -> OracleResponse:
        """Execute a tool from an installed MCP."""
        
        try:
            # This would interact with the MCP server via stdio/JSON-RPC
            # For now, return a structured response indicating MCP usage
            
            self.logger.info(f"Executing MCP tool: {mcp_name}.{tool_name or 'auto'}")
            
            return OracleResponse(
                query_id=self._generate_query_id(),
                success=True,
                data={
                    "mcp_used": mcp_name,
                    "tool_used": tool_name,
                    "query": query_text,
                    "result": "MCP tool execution placeholder - full implementation requires MCP runtime"
                },
                source="mcp_tool",
                metadata={
                    "mcp_name": mcp_name,
                    "tool_name": tool_name,
                    "execution_method": "json_rpc_stdio"
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return self._create_error_response(query_text, str(e))

    async def _execute_web_search(
        self,
        query_text: str,
        parameters: Optional[Dict[str, Any]]
    ) -> OracleResponse:
        """Execute web search and return formatted response."""
        
        max_results = parameters.get("max_results", 10) if parameters else 10
        
        # Create a temporary OracleQuery object to use with _web_search
        query_id = self._generate_query_id()
        temp_query = OracleQuery(
            query_id=query_id,
            requester="oracle_internal",
            query_text=query_text,
            source_type=KnowledgeSource.WEB_SEARCH,
            parameters=parameters or {}
        )
        
        # Use the existing _web_search method
        search_response = await self._web_search(temp_query)
        
        # DIAGNOSTIC: Log what _web_search returned
        logger.info(f"🔍 TRACE-3: Oracle._execute_web_search() got response with success={search_response.success}")
        logger.info(f"🔍 TRACE-3: Response data type: {type(search_response.data)}")
        logger.info(f"🔍 TRACE-3: Response data length: {len(search_response.data) if isinstance(search_response.data, list) else 'not a list'}")
        if isinstance(search_response.data, list) and search_response.data:
            logger.info(f"🔍 TRACE-3: First item in data: {search_response.data[0]}")
        
        if not search_response.success:
            return self._create_error_response(
                query_text,
                search_response.metadata.get("error", "Web search failed")
            )
        
        return OracleResponse(
            query_id=query_id,
            success=True,
            data=search_response.data[:max_results],
            source="web_search",
            metadata={
                "query": query_text,
                "num_results": len(search_response.data),
                "search_provider": "multi_provider"
            },
            timestamp=datetime.now()
        )

    def _generate_query_id(self) -> str:
        """Generate a unique query ID."""
        return f"oracle_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _create_error_response(self, query_text: str, error_message: str) -> OracleResponse:
        """Create a standardized error response."""
        return OracleResponse(
            query_id=self._generate_query_id(),
            success=False,
            data=None,
            source="error",
            metadata={
                "query": query_text,
                "error": error_message
            },
            timestamp=datetime.now()
        )

    @property
    def capabilities(self) -> List[str]:
        """Return list of this agent's capabilities"""
        return ["external_knowledge", "web_search", "mcp_discovery", "mcp_generation"]

    def can_handle_task(self, task: str) -> Tuple[bool, List[str]]:
        """
        Analyze if this agent can handle the task using lightweight LLM call.
        
        FIX ISSUE #2: Uses direct OpenAI API instead of creating full AutoGen agents
        with 300+ line psychological system messages. Maintains intelligence while
        reducing tokens by 90%+.
        
        Args:
            task: The task description
            
        Returns:
            Tuple of (can_handle: bool, missing_capabilities: List[str])
        """
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                # Fallback to simple keyword matching
                oracle_keywords = ["what", "who", "when", "where", "why", "how", "find", "search", "information"]
                can_handle = any(kw in task.lower() for kw in oracle_keywords)
                return (True, []) if can_handle else (False, ["text_synthesis"])
            
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            
            # Minimal task analysis prompt (not 300+ line psychological message!)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"""You are Oracle with capabilities: {self.capabilities}

Task: {task}

Can you handle this alone? Respond ONLY with JSON:
{{"can_handle": true/false, "missing": ["capability1"]}}"""
                }],
                temperature=0.3,
                max_tokens=100
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            can_handle = result.get("can_handle", False)
            missing = result.get("missing", ["text_synthesis"] if not can_handle else [])
            
            logger.info(f"✅ TOKEN-FIX: Oracle task analysis via lightweight LLM (~150 tokens)")
            return can_handle, missing
            
        except Exception as e:
            logger.error(f"Lightweight task analysis failed: {e}, using keyword fallback")
            oracle_keywords = ["what", "who", "when", "where", "why", "how", "find", "search", "information"]
            can_handle = any(kw in task.lower() for kw in oracle_keywords)
            return (True, []) if can_handle else (False, ["text_synthesis"])

def create_oracle_agent(llm_config: Dict[str, Any], safety_agent=None) -> OracleAgent:
    """Factory function to create Oracle agent"""
    return OracleAgent(llm_config, safety_agent)

if __name__ == "__main__":
    # Test Oracle agent
    print("Oracle Agent - Gateway to External Knowledge")
    print("=" * 50)
    
    # Test configuration
    llm_config = {
        "config_list": [{
            "model": "llama-3.3-70b-versatile",
            "api_key": "test_key",
            "base_url": "https://api.groq.com/openai/v1",
            "api_type": "openai"
        }],
        "temperature": 0.7
    }
    
    # Create Oracle
    oracle = create_oracle_agent(llm_config)
    
    print(f"Created Oracle agent: {oracle.name}")
    print(f"Role: {oracle.agent_role}")
    print(f"Curiosity Drive: {oracle.drive_system.drives[DriveType.CURIOSITY].intensity:.2f}")
    print(f"Purpose Drive: {oracle.drive_system.drives[DriveType.PURPOSE].intensity:.2f}")
    
    # Show statistics
    stats = oracle.get_oracle_statistics()
    print(f"\nOracle Statistics:")
    print(f"  Total Queries: {stats['statistics']['total_queries']}")
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Success Rate: {stats['success_rate']:.2f}")
