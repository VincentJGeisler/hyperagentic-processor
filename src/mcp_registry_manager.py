"""
MCP Registry Manager - Discovery System for MCP Servers

This module provides the MCPRegistryManager class that searches for existing MCP servers
across multiple registries (GitHub, npm, PyPI) and ranks them by relevance and trust score.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import os

logger = logging.getLogger("MCPRegistryManager")


@dataclass
class MCPPackage:
    """
    Represents an MCP server package discovered from a registry.
    """
    name: str
    description: str
    version: str
    source: str  # "github_official", "npm", "pypi", "github_community"
    repository_url: str
    install_command: str
    language: str  # "typescript", "python"
    trust_score: float
    downloads: int = 0
    stars: int = 0
    last_updated: Optional[str] = None
    compatibility: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "source": self.source,
            "repository_url": self.repository_url,
            "install_command": self.install_command,
            "language": self.language,
            "trust_score": self.trust_score,
            "downloads": self.downloads,
            "stars": self.stars,
            "last_updated": self.last_updated,
            "compatibility": self.compatibility,
            "metadata": self.metadata
        }


class MCPRegistryManager:
    """
    Manages MCP server discovery across multiple registries.
    
    Searches GitHub (official MCP org), npm registry, PyPI, and GitHub community
    for MCP servers, calculates trust scores, and ranks by relevance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MCP Registry Manager.
        
        Args:
            config: Configuration dict from mcp.json
        """
        self.config = config or {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Registry configurations
        self.registries = self.config.get("mcp_discovery", {}).get("registries", {})
        
        # GitHub configuration
        self.github_enabled = self.registries.get("github", {}).get("enabled", True)
        self.github_api_url = self.registries.get("github", {}).get("api_url", "https://api.github.com")
        self.github_token = os.getenv("GITHUB_TOKEN")
        
        # npm configuration
        self.npm_enabled = self.registries.get("npm", {}).get("enabled", True)
        self.npm_api_url = self.registries.get("npm", {}).get("api_url", "https://registry.npmjs.org")
        
        # PyPI configuration
        self.pypi_enabled = self.registries.get("pypi", {}).get("enabled", True)
        self.pypi_api_url = self.registries.get("pypi", {}).get("api_url", "https://pypi.org")
        
        # Trust scores
        self.trust_scores = {
            "github_official": 1.0,
            "npm": 0.9,
            "pypi": 0.8,
            "github_community": 0.6
        }
        
        logger.info("MCP Registry Manager initialized")
    
    async def initialize(self):
        """Initialize async components (HTTP session)."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("HTTP session initialized")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("HTTP session closed")
    
    async def discover_mcps(self, query: str, max_results: int = 10) -> List[MCPPackage]:
        """
        Discover MCP servers across all registries.
        
        Args:
            query: Search query (e.g., "pdf", "database", "github")
            max_results: Maximum number of results to return
            
        Returns:
            List of MCPPackage objects ranked by relevance and trust score
        """
        logger.info(f"Discovering MCP servers for query: {query}")
        
        if self.session is None:
            await self.initialize()
        
        # Search all registries in parallel
        search_tasks = []
        
        if self.github_enabled:
            search_tasks.append(self._search_github_official(query))
        
        if self.npm_enabled:
            search_tasks.append(self._search_npm_registry(query))
        
        if self.pypi_enabled:
            search_tasks.append(self._search_pypi(query))
        
        if self.github_enabled:
            search_tasks.append(self._search_github_community(query))
        
        # Execute searches in parallel
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during parallel search: {e}")
            results = []
        
        # Flatten results and handle exceptions
        all_packages = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Search task failed: {result}")
                continue
            if isinstance(result, list):
                all_packages.extend(result)
        
        # Deduplicate by repository URL
        unique_packages = self._deduplicate_packages(all_packages)
        
        # Rank by relevance
        ranked_packages = self._rank_by_relevance(unique_packages, query)
        
        # Return top results
        return ranked_packages[:max_results]
    
    async def _search_github_official(self, query: str) -> List[MCPPackage]:
        """
        Search official MCP organization on GitHub.
        
        Trust score: 100%
        """
        logger.info(f"Searching GitHub official MCP org for: {query}")
        packages = []
        
        try:
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            # Search modelcontextprotocol organization repos
            url = f"{self.github_api_url}/orgs/modelcontextprotocol/repos"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    repos = await response.json()
                    
                    for repo in repos:
                        # Filter by query match in name or description
                        name = repo.get("name", "").lower()
                        description = repo.get("description", "").lower()
                        query_lower = query.lower()
                        
                        if query_lower in name or query_lower in description or "mcp" in name:
                            # Extract package info
                            package_name = repo.get("name", "")
                            
                            # Determine language
                            language = self._determine_language(repo)
                            
                            # Create install command
                            if language == "typescript":
                                install_cmd = f"npx -y @modelcontextprotocol/{package_name}"
                            else:
                                install_cmd = f"uvx {package_name}"
                            
                            package = MCPPackage(
                                name=package_name,
                                description=repo.get("description", "No description"),
                                version="latest",
                                source="github_official",
                                repository_url=repo.get("html_url", ""),
                                install_command=install_cmd,
                                language=language,
                                trust_score=self.trust_scores["github_official"],
                                stars=repo.get("stargazers_count", 0),
                                last_updated=repo.get("updated_at"),
                                metadata={
                                    "full_name": repo.get("full_name"),
                                    "forks": repo.get("forks_count", 0),
                                    "open_issues": repo.get("open_issues_count", 0)
                                }
                            )
                            packages.append(package)
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Error searching GitHub official: {e}")
        
        logger.info(f"Found {len(packages)} packages from GitHub official")
        return packages
    
    async def _search_npm_registry(self, query: str) -> List[MCPPackage]:
        """
        Search npm registry for MCP packages.
        
        Trust score: 90%
        """
        logger.info(f"Searching npm registry for: {query}")
        packages = []
        
        try:
            # Search npm registry
            search_query = f"{query} mcp model-context-protocol"
            url = f"https://registry.npmjs.org/-/v1/search?text={search_query}&size=20"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    objects = data.get("objects", [])
                    
                    for obj in objects:
                        pkg = obj.get("package", {})
                        name = pkg.get("name", "")
                        
                        # Filter for MCP-related packages
                        if "mcp" in name.lower() or "model-context-protocol" in name.lower():
                            package = MCPPackage(
                                name=name,
                                description=pkg.get("description", "No description"),
                                version=pkg.get("version", "latest"),
                                source="npm",
                                repository_url=pkg.get("links", {}).get("repository", ""),
                                install_command=f"npx -y {name}",
                                language="typescript",
                                trust_score=self.trust_scores["npm"],
                                downloads=obj.get("score", {}).get("detail", {}).get("popularity", 0) * 10000,
                                metadata={
                                    "npm_score": obj.get("score", {}).get("final", 0),
                                    "quality": obj.get("score", {}).get("detail", {}).get("quality", 0),
                                    "maintenance": obj.get("score", {}).get("detail", {}).get("maintenance", 0)
                                }
                            )
                            packages.append(package)
                else:
                    logger.warning(f"npm registry returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Error searching npm registry: {e}")
        
        logger.info(f"Found {len(packages)} packages from npm")
        return packages
    
    async def _search_pypi(self, query: str) -> List[MCPPackage]:
        """
        Search PyPI for MCP packages.
        
        Trust score: 80%
        """
        logger.info(f"Searching PyPI for: {query}")
        packages = []
        
        try:
            # Search PyPI (using their simple search)
            # Note: PyPI doesn't have a great search API, so we search for common patterns
            search_terms = [
                f"mcp-server-{query}",
                f"mcp_{query}",
                f"{query}-mcp"
            ]
            
            for search_term in search_terms:
                url = f"{self.pypi_api_url}/pypi/{search_term}/json"
                
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            info = data.get("info", {})
                            
                            package = MCPPackage(
                                name=info.get("name", search_term),
                                description=info.get("summary", "No description"),
                                version=info.get("version", "latest"),
                                source="pypi",
                                repository_url=info.get("project_urls", {}).get("Source", ""),
                                install_command=f"uvx {info.get('name', search_term)}",
                                language="python",
                                trust_score=self.trust_scores["pypi"],
                                downloads=0,  # PyPI API doesn't provide download counts easily
                                metadata={
                                    "author": info.get("author", ""),
                                    "license": info.get("license", ""),
                                    "requires_python": info.get("requires_python", "")
                                }
                            )
                            packages.append(package)
                except aiohttp.ClientResponseError:
                    # Package doesn't exist, continue
                    continue
        
        except Exception as e:
            logger.error(f"Error searching PyPI: {e}")
        
        logger.info(f"Found {len(packages)} packages from PyPI")
        return packages
    
    async def _search_github_community(self, query: str) -> List[MCPPackage]:
        """
        Search GitHub community repositories for MCP servers.
        
        Trust score: 60% (varies by stars/activity)
        """
        logger.info(f"Searching GitHub community for: {query}")
        packages = []
        
        try:
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            # Search GitHub repos with MCP-related topics
            search_query = f"{query} mcp server topic:mcp-server topic:model-context-protocol"
            url = f"{self.github_api_url}/search/repositories?q={search_query}&sort=stars&order=desc&per_page=10"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("items", [])
                    
                    for repo in items:
                        # Skip official repos (already covered)
                        if repo.get("full_name", "").startswith("modelcontextprotocol/"):
                            continue
                        
                        name = repo.get("name", "")
                        language = self._determine_language(repo)
                        
                        # Adjust trust score based on stars and activity
                        base_trust = self.trust_scores["github_community"]
                        stars = repo.get("stargazers_count", 0)
                        trust_boost = min(stars / 100, 0.3)  # Up to 0.3 boost for popular repos
                        trust_score = min(base_trust + trust_boost, 0.95)
                        
                        # Create install command
                        if language == "typescript":
                            install_cmd = f"npx -y {repo.get('full_name', name)}"
                        else:
                            install_cmd = f"git clone {repo.get('clone_url')} && cd {name} && pip install -e ."
                        
                        package = MCPPackage(
                            name=name,
                            description=repo.get("description", "No description"),
                            version="latest",
                            source="github_community",
                            repository_url=repo.get("html_url", ""),
                            install_command=install_cmd,
                            language=language,
                            trust_score=trust_score,
                            stars=stars,
                            last_updated=repo.get("updated_at"),
                            metadata={
                                "full_name": repo.get("full_name"),
                                "forks": repo.get("forks_count", 0),
                                "open_issues": repo.get("open_issues_count", 0),
                                "license": repo.get("license", {}).get("name", "Unknown") if repo.get("license") else "Unknown"
                            }
                        )
                        packages.append(package)
                else:
                    logger.warning(f"GitHub search returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Error searching GitHub community: {e}")
        
        logger.info(f"Found {len(packages)} packages from GitHub community")
        return packages
    
    def _determine_language(self, repo: Dict[str, Any]) -> str:
        """Determine the primary language of a repository."""
        language = repo.get("language", "").lower()
        
        if language in ["typescript", "javascript"]:
            return "typescript"
        elif language in ["python"]:
            return "python"
        else:
            # Default to typescript for MCP servers
            return "typescript"
    
    def _deduplicate_packages(self, packages: List[MCPPackage]) -> List[MCPPackage]:
        """
        Deduplicate packages by repository URL.
        Keeps the one with highest trust score.
        """
        seen_urls = {}
        
        for package in packages:
            url = package.repository_url
            if not url:
                # No URL, include it
                continue
            
            if url not in seen_urls:
                seen_urls[url] = package
            else:
                # Keep the one with higher trust score
                if package.trust_score > seen_urls[url].trust_score:
                    seen_urls[url] = package
        
        return list(seen_urls.values())
    
    def _rank_by_relevance(self, packages: List[MCPPackage], query: str) -> List[MCPPackage]:
        """
        Rank packages by relevance to query and trust score.
        
        Scoring factors:
        - Query match in name (high weight)
        - Query match in description (medium weight)
        - Trust score (high weight)
        - Stars/popularity (low weight)
        """
        query_lower = query.lower()
        
        def calculate_relevance_score(package: MCPPackage) -> float:
            score = 0.0
            
            # Name match (0-100 points)
            name_lower = package.name.lower()
            if query_lower == name_lower:
                score += 100
            elif query_lower in name_lower:
                score += 70
            elif any(word in name_lower for word in query_lower.split()):
                score += 40
            
            # Description match (0-50 points)
            desc_lower = package.description.lower()
            if query_lower in desc_lower:
                score += 50
            elif any(word in desc_lower for word in query_lower.split()):
                score += 25
            
            # Trust score (0-100 points)
            score += package.trust_score * 100
            
            # Popularity (0-20 points)
            if package.stars > 0:
                score += min(package.stars / 10, 20)
            elif package.downloads > 0:
                score += min(package.downloads / 1000, 20)
            
            return score
        
        # Calculate scores and sort
        scored_packages = [(package, calculate_relevance_score(package)) for package in packages]
        scored_packages.sort(key=lambda x: x[1], reverse=True)
        
        return [package for package, score in scored_packages]
    
    def _calculate_trust_score(self, source: str, metadata: Dict[str, Any]) -> float:
        """
        Calculate trust score for a package.
        
        Base trust scores:
        - Official GitHub: 100%
        - npm: 90%
        - PyPI: 80%
        - Community: 60%
        
        Adjusted by metadata (stars, maintenance, etc.)
        """
        base_score = self.trust_scores.get(source, 0.5)
        
        # Adjust based on metadata
        if source == "github_community":
            stars = metadata.get("stars", 0)
            # Boost up to 0.3 for popular repos
            boost = min(stars / 100, 0.3)
            return min(base_score + boost, 0.95)
        
        elif source == "npm":
            quality = metadata.get("quality", 0)
            maintenance = metadata.get("maintenance", 0)
            # Adjust based on npm scores
            adjustment = (quality + maintenance) / 2 * 0.1
            return min(base_score + adjustment, 1.0)
        
        return base_score


# Factory function
def create_mcp_registry_manager(config: Optional[Dict[str, Any]] = None) -> MCPRegistryManager:
    """Factory function to create MCP Registry Manager."""
    return MCPRegistryManager(config)


if __name__ == "__main__":
    # Test the registry manager
    import asyncio
    
    async def test():
        manager = MCPRegistryManager()
        await manager.initialize()
        
        try:
            packages = await manager.discover_mcps("pdf", max_results=5)
            print(f"Found {len(packages)} MCP packages:")
            for pkg in packages:
                print(f"  - {pkg.name} ({pkg.source}) - Trust: {pkg.trust_score:.2f}")
                print(f"    {pkg.description}")
                print(f"    Install: {pkg.install_command}")
                print()
        finally:
            await manager.close()
    
    asyncio.run(test())
