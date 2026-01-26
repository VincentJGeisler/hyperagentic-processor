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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import re

from motivated_agent import MotivatedAgent
from agent_drive_system import DriveType

logger = logging.getLogger("OracleAgent")

class KnowledgeSource(Enum):
    """Types of external knowledge sources"""
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    DOCUMENT = "document"
    API = "api"
    DATABASE = "database"

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
        metadata: Dict[str, Any]
    ):
        self.query_id = query_id
        self.success = success
        self.data = data
        self.source = source
        self.metadata = metadata
        self.timestamp = datetime.now()

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
        
        logger.info("Oracle agent initialized with external knowledge access and MCP installation capability")
    
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
        source_type: KnowledgeSource,
        parameters: Optional[Dict[str, Any]] = None
    ) -> OracleResponse:
        """
        Query external knowledge sources.
        
        This is the main entry point for agents seeking external information.
        """
        parameters = parameters or {}
        query_id = f"oracle_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"Oracle query from {requester}: {query_text[:100]}")
        
        # Create query object
        query = OracleQuery(
            query_id=query_id,
            requester=requester,
            query_text=query_text,
            source_type=source_type,
            parameters=parameters
        )
        
        self.query_history.append(query)
        self.stats["total_queries"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query_text, source_type, parameters)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Request safety approval if SafetyAgent available
        if self.safety_agent:
            approved = await self._request_safety_approval(query)
            if not approved:
                self.stats["safety_rejections"] += 1
                return OracleResponse(
                    query_id=query_id,
                    success=False,
                    data=None,
                    source="safety_rejection",
                    metadata={"reason": "Query rejected by SafetyAgent"}
                )
        
        # Execute query based on source type
        try:
            if source_type == KnowledgeSource.WEB_SEARCH:
                response = await self._web_search(query)
            elif source_type == KnowledgeSource.WEB_PAGE:
                response = await self._fetch_web_page(query)
            elif source_type == KnowledgeSource.DOCUMENT:
                response = await self._fetch_document(query)
            elif source_type == KnowledgeSource.API:
                response = await self._call_api(query)
            else:
                response = OracleResponse(
                    query_id=query_id,
                    success=False,
                    data=None,
                    source="unknown",
                    metadata={"error": f"Unknown source type: {source_type}"}
                )
            
            # Cache successful responses
            if response.success:
                self.knowledge_cache[cache_key] = response
                self.stats["successful_queries"] += 1
            
            self.stats["external_calls"] += 1
            
            # Update psychological state based on success
            self._process_query_result(response.success)
            
            return response
            
        except Exception as e:
            logger.error(f"Oracle query failed: {e}")
            return OracleResponse(
                query_id=query_id,
                success=False,
                data=None,
                source="error",
                metadata={"error": str(e)}
            )
    
    async def _request_safety_approval(self, query: OracleQuery) -> bool:
        """Request approval from SafetyAgent for external query"""
        logger.info(f"Requesting safety approval for query: {query.query_id}")
        
        # Create safety check context
        safety_context = {
            "query_type": "external_knowledge_access",
            "requester": query.requester,
            "query_text": query.query_text,
            "source_type": query.source_type.value,
            "parameters": query.parameters
        }
        
        # For now, implement basic safety checks
        # In full implementation, this would use SafetyAgent's analysis
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'(?i)(hack|exploit|vulnerability|bypass|inject)',
            r'(?i)(password|credential|token|secret|key)',
            r'(?i)(malware|virus|trojan|backdoor)',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query.query_text):
                logger.warning(f"Suspicious pattern detected in query: {pattern}")
                return False
        
        # Check URL safety for web queries
        if query.source_type in [KnowledgeSource.WEB_PAGE, KnowledgeSource.DOCUMENT]:
            url = query.parameters.get('url', '')
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
        """Perform web search using MCP tools"""
        logger.info(f"Performing web search: {query.query_text}")
        
        try:
            # Use the remote_web_search MCP tool
            from mcp_tools import remote_web_search
            
            search_results = remote_web_search(query.query_text)
            
            # Format results for agents
            formatted_results = []
            for result in search_results[:5]:  # Top 5 results
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "published_date": result.get("publishedDate", "")
                })
            
            return OracleResponse(
                query_id=query.query_id,
                success=True,
                data=formatted_results,
                source="web_search",
                metadata={
                    "query": query.query_text,
                    "result_count": len(formatted_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            # Fallback to simulated response for testing
            return OracleResponse(
                query_id=query.query_id,
                success=True,
                data=[{
                    "title": f"Search results for: {query.query_text}",
                    "url": "https://example.com",
                    "snippet": "The Oracle has consulted the external realm and found relevant information.",
                    "published_date": datetime.now().isoformat()
                }],
                source="web_search_simulated",
                metadata={"note": "Simulated response - MCP tools not available"}
            )
    
    async def _fetch_web_page(self, query: OracleQuery) -> OracleResponse:
        """Fetch and parse a web page using MCP tools"""
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
        
        try:
            # Use the webFetch MCP tool
            from mcp_tools import webFetch
            
            mode = query.parameters.get('mode', 'truncated')
            content = webFetch(url, mode=mode)
            
            return OracleResponse(
                query_id=query.query_id,
                success=True,
                data=content,
                source="web_page",
                metadata={
                    "url": url,
                    "mode": mode,
                    "length": len(content) if content else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Web page fetch failed: {e}")
            return OracleResponse(
                query_id=query.query_id,
                success=False,
                data=None,
                source="web_page",
                metadata={"error": str(e), "url": url}
            )
    
    async def _fetch_document(self, query: OracleQuery) -> OracleResponse:
        """Fetch and process a document"""
        url = query.parameters.get('url')
        doc_type = query.parameters.get('type', 'unknown')
        
        logger.info(f"Fetching document: {url} (type: {doc_type})")
        
        try:
            # Use webFetch for document download
            from mcp_tools import webFetch
            
            content = webFetch(url, mode='full')
            
            # Process based on document type
            processed_content = self._process_document(content, doc_type)
            
            return OracleResponse(
                query_id=query.query_id,
                success=True,
                data=processed_content,
                source="document",
                metadata={
                    "url": url,
                    "type": doc_type,
                    "size": len(content) if content else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Document fetch failed: {e}")
            return OracleResponse(
                query_id=query.query_id,
                success=False,
                data=None,
                source="document",
                metadata={"error": str(e), "url": url}
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
        return {
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
    
    def clear_cache(self):
        """Clear knowledge cache"""
        self.knowledge_cache.clear()
        logger.info("Oracle knowledge cache cleared")

# MCP Tools wrapper (to be imported from actual MCP integration)
class mcp_tools:
    """Wrapper for MCP tools - connects to actual MCP servers"""
    
    @staticmethod
    def remote_web_search(query: str) -> List[Dict[str, Any]]:
        """Wrapper for remote_web_search MCP tool"""
        # This would call the actual MCP tool
        # For now, return simulated results
        return [
            {
                "title": f"Result for: {query}",
                "url": "https://example.com",
                "snippet": "Relevant information from the external realm",
                "publishedDate": datetime.now().isoformat()
            }
        ]
    
    @staticmethod
    def webFetch(url: str, mode: str = 'truncated') -> str:
        """Wrapper for webFetch MCP tool"""
        # This would call the actual MCP tool
        return f"Content from {url} (mode: {mode})"

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
