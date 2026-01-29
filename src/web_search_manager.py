"""
Web Search Manager - Real Web Search Implementation

Provides real web search capabilities using multiple providers:
- Brave Search API (primary)
- Tavily AI Search (secondary)
- SerpAPI (fallback)

Features:
- Multi-provider with automatic fallback
- LRU cache with TTL
- Rate limiting per provider
- Connection pooling
- Result normalization
- Graceful degradation
"""

import os
import logging
import time
import asyncio
import hashlib
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import aiohttp
from cachetools import TTLCache

logger = logging.getLogger("WebSearchManager")


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    source: str = "web_search"
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests_per_hour: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_hour: Maximum requests allowed per hour
        """
        self.max_requests_per_hour = max_requests_per_hour
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Attempt to acquire rate limit slot.
        
        Returns:
            True if request allowed, False if rate limited
        """
        async with self.lock:
            now = time.time()
            hour_ago = now - 3600
            
            # Remove requests older than 1 hour
            self.requests = [req_time for req_time in self.requests if req_time > hour_ago]
            
            if len(self.requests) >= self.max_requests_per_hour:
                logger.warning(f"Rate limit exceeded: {len(self.requests)}/{self.max_requests_per_hour}")
                return False
            
            self.requests.append(now)
            return True
    
    def get_remaining(self) -> int:
        """Get remaining requests in current hour."""
        now = time.time()
        hour_ago = now - 3600
        recent_requests = sum(1 for req_time in self.requests if req_time > hour_ago)
        return max(0, self.max_requests_per_hour - recent_requests)


class WebSearchManager:
    """
    Multi-provider web search manager with caching and rate limiting.
    
    Supports multiple search providers with automatic fallback:
    1. Brave Search API (primary)
    2. Tavily AI Search (secondary)
    3. SerpAPI (tertiary/fallback)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web search manager.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Initialize HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize cache (1 hour TTL by default)
        cache_ttl = self.config.get("cache", {}).get("ttl_seconds", 3600)
        cache_max_size = self.config.get("cache", {}).get("max_entries", 1000)
        self.cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl)
        self.cache_lock = asyncio.Lock()
        
        # Initialize rate limiters for each provider
        self.rate_limiters: Dict[str, RateLimiter] = {}
        providers = self.config.get("providers", {})
        for provider_name, provider_config in providers.items():
            rate_limit = provider_config.get("rate_limit", 100)
            self.rate_limiters[provider_name] = RateLimiter(rate_limit)
        
        # Statistics
        self.stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_calls": defaultdict(int),
            "provider_errors": defaultdict(int),
            "provider_fallbacks": 0
        }
        
        logger.info("WebSearchManager initialized")
    
    async def initialize(self):
        """Initialize HTTP session."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=10,              # Max 10 concurrent connections
                limit_per_host=3,      # Max 3 per host
                ttl_dns_cache=300,     # 5 min DNS cache
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            logger.info("HTTP session initialized")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("HTTP session closed")
    
    async def _ensure_session_valid(self):
        """Ensure HTTP session is valid and not closed."""
        if self.session is None or self.session.closed:
            logger.info("Session is None or closed, reinitializing...")
            await self.initialize()
    
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Execute web search with multi-provider fallback.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        self.stats["total_searches"] += 1
        
        # Check cache first
        cached_results = await self._get_cached(query, num_results)
        if cached_results is not None:
            self.stats["cache_hits"] += 1
            logger.info(f"Cache hit for query: {query[:50]}")
            return cached_results
        
        self.stats["cache_misses"] += 1
        
        # Ensure session is initialized
        if self.session is None:
            await self.initialize()
        
        # Get enabled providers sorted by priority
        providers = self._get_enabled_providers()
        
        if not providers:
            logger.error("No search providers available")
            return []
        
        # Try each provider in order
        for provider_name, provider_config in providers:
            try:
                # Check rate limit
                rate_limiter = self.rate_limiters.get(provider_name)
                if rate_limiter and not await rate_limiter.acquire():
                    logger.warning(f"Rate limit exceeded for {provider_name}, trying next provider")
                    continue
                
                # Execute search
                logger.info(f"Searching with provider: {provider_name}")
                results = await self._search_with_provider(
                    provider_name,
                    provider_config,
                    query,
                    num_results
                )
                
                if results:
                    # Cache successful results
                    await self._cache_results(query, num_results, results)
                    self.stats["provider_calls"][provider_name] += 1
                    # DIAGNOSTIC: Log what we're returning
                    logger.info(f"🔍 TRACE-1: WebSearchManager.search() returning {len(results)} results")
                    if results:
                        logger.info(f"🔍 TRACE-1: First result type: {type(results[0])}, title: {results[0].title[:50] if results[0].title else 'None'}")
                    return results
                    
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                self.stats["provider_errors"][provider_name] += 1
                self.stats["provider_fallbacks"] += 1
                # Continue to next provider
        
        # All providers failed
        logger.error(f"All providers failed for query: {query}")
        return []
    
    def _get_enabled_providers(self) -> List[tuple]:
        """
        Get enabled providers sorted by priority.
        
        Returns:
            List of (provider_name, provider_config) tuples
        """
        providers = self.config.get("providers", {})
        enabled = []
        
        logger.info(f"🔍 DEBUG: Checking providers, total count: {len(providers)}")
        logger.info(f"🔍 DEBUG: Full config received: {json.dumps(self.config, indent=2)}")
        
        for name, config in providers.items():
            is_enabled = config.get("enabled", False)
            # Check api_key_available flag first, then fall back to checking environment directly
            has_api_key = config.get("api_key_available", False)
            if not has_api_key and config.get("api_key_env"):
                # Check environment variable directly as fallback
                api_key_env = config.get("api_key_env")
                has_api_key = bool(os.getenv(api_key_env)) if api_key_env else False
            logger.info(f"🔍 DEBUG: Provider '{name}': enabled={is_enabled}, api_key_available={has_api_key}")
            
            if is_enabled and has_api_key:
                enabled.append((name, config))
                logger.info(f"✅ Provider '{name}' is ENABLED and will be used")
            else:
                logger.warning(f"❌ Provider '{name}' SKIPPED: enabled={is_enabled}, api_key_available={has_api_key}")
        
        # Sort by priority (lower number = higher priority)
        enabled.sort(key=lambda x: x[1].get("priority", 999))
        
        logger.info(f"🔍 DEBUG: Final enabled providers: {[name for name, _ in enabled]}")
        return enabled
    
    async def _search_with_provider(
        self,
        provider_name: str,
        provider_config: Dict[str, Any],
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """
        Execute search with specific provider.
        
        Args:
            provider_name: Name of provider (brave, tavily, serpapi)
            provider_config: Provider configuration
            query: Search query
            num_results: Number of results requested
            
        Returns:
            List of normalized SearchResult objects
        """
        if provider_name == "brave":
            return await self._search_brave(query, num_results)
        elif provider_name == "tavily":
            return await self._search_tavily(query, num_results)
        elif provider_name == "serpapi":
            return await self._search_serpapi(query, num_results)
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return []
    
    async def _search_brave(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using Brave Search API.
        
        API Details:
        - Endpoint: https://api.search.brave.com/res/v1/web/search
        - Headers: X-Subscription-Token: {api_key}
        - Query params: q={query}&count={num_results}
        """
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            raise ValueError("BRAVE_API_KEY not found in environment")
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": num_results
        }
        
        # Try with current session, recreate if event loop is closed
        try:
            await self._ensure_session_valid()
            async with self.session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Extract results from Brave API response
                raw_results = data.get("web", {}).get("results", [])
                return self._normalize_results(raw_results, "brave")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.warning("Event loop closed, recreating session...")
                # Force close old session
                if self.session:
                    try:
                        await self.session.close()
                    except:
                        pass
                self.session = None
                # Reinitialize with new loop
                await self.initialize()
                # Retry the request
                async with self.session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    raw_results = data.get("web", {}).get("results", [])
                    return self._normalize_results(raw_results, "brave")
            else:
                raise
    
    async def _search_tavily(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using Tavily AI Search API.
        
        API Details:
        - Endpoint: https://api.tavily.com/search
        - Method: POST
        - Body: {"api_key": key, "query": query, "max_results": num}
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        
        # Ensure session is valid (not closed)
        await self._ensure_session_valid()
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": num_results
        }
        
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Extract results from Tavily API response
            raw_results = data.get("results", [])
            return self._normalize_results(raw_results, "tavily")
    
    async def _search_serpapi(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using SerpAPI.
        
        API Details:
        - Endpoint: https://serpapi.com/search
        - Query params: q={query}&api_key={key}&num={num_results}
        """
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            raise ValueError("SERPAPI_KEY not found in environment")
        
        # Ensure session is valid (not closed)
        await self._ensure_session_valid()
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google"
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Extract results from SerpAPI response
            raw_results = data.get("organic_results", [])
            return self._normalize_results(raw_results, "serpapi")
    
    def _normalize_results(
        self,
        raw_results: List[Dict],
        provider: str
    ) -> List[SearchResult]:
        """
        Normalize results from different providers to common format.
        
        Args:
            raw_results: Raw results from provider
            provider: Provider name (brave, tavily, serpapi)
            
        Returns:
            List of normalized SearchResult objects
        """
        normalized = []
        
        for result in raw_results:
            try:
                # Extract fields based on provider
                if provider == "brave":
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("description", ""),
                        published_date=result.get("published_date"),
                        source=provider,
                        relevance_score=result.get("relevance_score", 0.0),
                        metadata={"provider": provider}
                    )
                elif provider == "tavily":
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("content", ""),
                        published_date=result.get("published_date"),
                        source=provider,
                        relevance_score=result.get("score", 0.0),
                        metadata={"provider": provider}
                    )
                elif provider == "serpapi":
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        published_date=result.get("date"),
                        source=provider,
                        relevance_score=result.get("position", 0) / 10.0,  # Convert position to score
                        metadata={"provider": provider, "position": result.get("position")}
                    )
                else:
                    continue
                
                normalized.append(search_result)
                
            except Exception as e:
                logger.warning(f"Failed to normalize result from {provider}: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} results from {provider}")
        return normalized
    
    def _generate_cache_key(self, query: str, num_results: int) -> str:
        """Generate cache key for query."""
        cache_data = f"{query}:{num_results}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    async def _get_cached(
        self,
        query: str,
        num_results: int
    ) -> Optional[List[SearchResult]]:
        """
        Get cached search results.
        
        Args:
            query: Search query
            num_results: Number of results
            
        Returns:
            Cached results or None if not found/expired
        """
        cache_key = self._generate_cache_key(query, num_results)
        
        async with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        return None
    
    async def _cache_results(
        self,
        query: str,
        num_results: int,
        results: List[SearchResult]
    ):
        """
        Cache search results.
        
        Args:
            query: Search query
            num_results: Number of results
            results: Search results to cache
        """
        cache_key = self._generate_cache_key(query, num_results)
        
        async with self.cache_lock:
            self.cache[cache_key] = results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search manager statistics."""
        return {
            "total_searches": self.stats["total_searches"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["total_searches"])
            ),
            "provider_calls": dict(self.stats["provider_calls"]),
            "provider_errors": dict(self.stats["provider_errors"]),
            "provider_fallbacks": self.stats["provider_fallbacks"],
            "cache_size": len(self.cache),
            "rate_limits": {
                name: limiter.get_remaining()
                for name, limiter in self.rate_limiters.items()
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


if __name__ == "__main__":
    # Test web search manager
    import asyncio
    
    async def test_search():
        print("Web Search Manager Test")
        print("=" * 50)
        
        # Test configuration
        config = {
            "providers": {
                "brave": {
                    "enabled": True,
                    "priority": 1,
                    "api_key_env": "BRAVE_API_KEY",
                    "rate_limit": 2000,
                    "api_key_available": bool(os.getenv("BRAVE_API_KEY"))
                },
                "tavily": {
                    "enabled": True,
                    "priority": 2,
                    "api_key_env": "TAVILY_API_KEY",
                    "rate_limit": 1000,
                    "api_key_available": bool(os.getenv("TAVILY_API_KEY"))
                },
                "serpapi": {
                    "enabled": True,
                    "priority": 3,
                    "api_key_env": "SERPAPI_KEY",
                    "rate_limit": 100,
                    "api_key_available": bool(os.getenv("SERPAPI_KEY"))
                }
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 3600,
                "max_entries": 1000
            }
        }
        
        async with WebSearchManager(config) as manager:
            # Test search
            query = "artificial intelligence recent developments"
            print(f"\nSearching for: {query}")
            
            try:
                results = await manager.search(query, num_results=5)
                
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.title}")
                    print(f"   URL: {result.url}")
                    print(f"   Snippet: {result.snippet[:100]}...")
                    print(f"   Source: {result.source}")
                
                # Test cache
                print("\n\nTesting cache (same query)...")
                cached_results = await manager.search(query, num_results=5)
                print(f"Found {len(cached_results)} cached results")
                
                # Show statistics
                stats = manager.get_statistics()
                print(f"\n\nStatistics:")
                print(f"  Total searches: {stats['total_searches']}")
                print(f"  Cache hits: {stats['cache_hits']}")
                print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
                print(f"  Provider calls: {stats['provider_calls']}")
                
            except Exception as e:
                print(f"Search failed: {e}")
                print(f"Note: API keys must be set in environment variables")
    
    asyncio.run(test_search())
