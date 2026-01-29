"""
Agent Registry System

This module provides a centralized registry for tracking agent capabilities.
Enables dynamic agent discovery and intelligent routing based on capabilities
rather than hardcoded keyword matching.

The registry tracks:
- Agent instances
- Agent capabilities (what they can do)
- Agent descriptions (what they're for)
- Agent metadata (status, performance stats, etc.)
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger("AgentRegistry")


@dataclass
class AgentRegistration:
    """Represents a registered agent with its capabilities"""
    name: str
    capabilities: List[str]
    description: str
    agent_instance: Any
    registered_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "description": self.description,
            "registered_at": self.registered_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata
        }


class AgentRegistry:
    """
    Centralized registry for tracking agent capabilities.
    
    This replaces the brittle keyword-based routing with a flexible
    capability-based system. Agents register their capabilities, and
    the orchestrator can query for agents that can handle specific tasks.
    
    Example:
        >>> registry = AgentRegistry()
        >>> registry.register_agent(
        ...     name="oracle",
        ...     capabilities=["external_knowledge", "web_search"],
        ...     description="Gateway to external knowledge",
        ...     agent_instance=oracle_agent
        ... )
        >>> agents = registry.get_agents_by_capability("web_search")
        >>> # Returns ["oracle"]
    """
    
    def __init__(self):
        """Initialize the agent registry"""
        self.agents: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = {}
        logger.info("Agent registry initialized")
    
    def register_agent(
        self,
        name: str,
        capabilities: List[str],
        description: str,
        agent_instance: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an agent with its capabilities.
        
        Args:
            name: Unique agent identifier (e.g., "oracle", "tool_creator")
            capabilities: List of capabilities the agent provides
            description: Human-readable description of the agent
            agent_instance: The actual agent object
            metadata: Optional metadata (status, version, etc.)
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Check if agent already registered
            if name in self.agents:
                logger.warning(f"Agent '{name}' already registered, updating registration")
                # Unregister old capabilities
                self._unindex_agent(name)
            
            # Create registration
            registration = AgentRegistration(
                name=name,
                capabilities=capabilities,
                description=description,
                agent_instance=agent_instance,
                metadata=metadata or {}
            )
            
            # Store registration
            self.agents[name] = registration
            
            # Index capabilities
            self._index_capabilities(name, capabilities)
            
            logger.info(f"Registered agent '{name}' with capabilities: {capabilities}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent '{name}': {e}")
            return False
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            name: Agent identifier to unregister
        
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if name not in self.agents:
                logger.warning(f"Agent '{name}' not found in registry")
                return False
            
            # Remove from capability index
            self._unindex_agent(name)
            
            # Remove from agents
            del self.agents[name]
            
            logger.info(f"Unregistered agent '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent '{name}': {e}")
            return False
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """
        Find all agents that have a specific capability.
        
        Args:
            capability: The capability to search for
        
        Returns:
            List of agent names that have this capability
        """
        if capability not in self.capability_index:
            logger.debug(f"No agents found for capability '{capability}'")
            return []
        
        # Filter by active status
        agent_names = [
            name for name in self.capability_index[capability]
            if self.agents[name].status == "active"
        ]
        
        logger.debug(f"Found {len(agent_names)} agents for capability '{capability}': {agent_names}")
        return agent_names
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """
        Return all agents and their capabilities.
        
        Returns:
            Dictionary mapping agent names to their capability lists
        """
        return {
            name: registration.capabilities
            for name, registration in self.agents.items()
        }
    
    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance by name.
        
        Args:
            name: Agent identifier
        
        Returns:
            Agent instance or None if not found
        """
        if name not in self.agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return None
        
        return self.agents[name].agent_instance
    
    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an agent.
        
        Args:
            name: Agent identifier
        
        Returns:
            Dictionary with agent information or None if not found
        """
        if name not in self.agents:
            return None
        
        return self.agents[name].to_dict()
    
    def find_capable_agents(self, required_capabilities: List[str]) -> List[str]:
        """
        Find agents that have ALL of the required capabilities.
        
        Args:
            required_capabilities: List of capabilities needed
        
        Returns:
            List of agent names that have all required capabilities
        """
        if not required_capabilities:
            return []
        
        # Start with agents having the first capability
        candidate_agents = set(self.get_agents_by_capability(required_capabilities[0]))
        
        # Intersect with agents having each subsequent capability
        for capability in required_capabilities[1:]:
            capable_agents = set(self.get_agents_by_capability(capability))
            candidate_agents = candidate_agents.intersection(capable_agents)
        
        result = list(candidate_agents)
        logger.debug(f"Found {len(result)} agents with all capabilities {required_capabilities}: {result}")
        return result
    
    def find_best_agent_for_task(self, task_description: str) -> Optional[str]:
        """
        Analyze a task description and find the best agent to handle it.
        
        This method uses keyword matching and capability analysis to
        determine which agent is most suitable for a given task.
        
        Args:
            task_description: Description of the task to be performed
        
        Returns:
            Name of the best agent, or None if no suitable agent found
        """
        task_lower = task_description.lower()
        
        # Capability keywords mapping
        capability_keywords = {
            "external_knowledge": ["what", "who", "when", "where", "why", "how", "search", "find", "lookup", "query"],
            "web_search": ["web", "search", "online", "internet", "google"],
            "code_generation": ["create", "build", "generate", "write", "implement", "develop", "code"],
            "tool_creation": ["tool", "function", "script", "program"],
            "text_synthesis": ["explain", "summarize", "describe", "answer"],
            "security_analysis": ["safe", "secure", "analyze", "check", "validate", "security"],
            "performance_evaluation": ["evaluate", "assess", "grade", "measure", "performance", "quality"],
            "mcp_discovery": ["discover", "mcp", "capability", "server"],
            "mcp_generation": ["generate mcp", "create mcp"]
        }
        
        # Score each capability based on keyword matches
        capability_scores = {}
        for capability, keywords in capability_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                capability_scores[capability] = score
        
        if not capability_scores:
            return None
        
        # Find the capability with highest score
        best_capability = max(capability_scores, key=capability_scores.get)
        
        # Get agents with this capability
        capable_agents = self.get_agents_by_capability(best_capability)
        
        if not capable_agents:
            return None
        
        # Return the first capable agent (could be enhanced with priority system)
        return capable_agents[0]
    
    def update_agent_status(self, name: str, status: str) -> bool:
        """
        Update an agent's status.
        
        Args:
            name: Agent identifier
            status: New status (e.g., "active", "busy", "disabled")
        
        Returns:
            True if update successful, False otherwise
        """
        if name not in self.agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return False
        
        old_status = self.agents[name].status
        self.agents[name].status = status
        logger.info(f"Updated agent '{name}' status: {old_status} -> {status}")
        return True
    
    def update_agent_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an agent's metadata.
        
        Args:
            name: Agent identifier
            metadata: Metadata to update/add
        
        Returns:
            True if update successful, False otherwise
        """
        if name not in self.agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return False
        
        self.agents[name].metadata.update(metadata)
        logger.debug(f"Updated metadata for agent '{name}'")
        return True
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent registry.
        
        Returns:
            Dictionary with registry statistics
        """
        total_agents = len(self.agents)
        active_agents = sum(1 for a in self.agents.values() if a.status == "active")
        total_capabilities = len(self.capability_index)
        
        # Most common capabilities
        capability_counts = {
            cap: len(agents) for cap, agents in self.capability_index.items()
        }
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_capabilities": total_capabilities,
            "capability_distribution": capability_counts,
            "agents": {
                name: {
                    "status": reg.status,
                    "capabilities_count": len(reg.capabilities),
                    "registered_at": reg.registered_at.isoformat()
                }
                for name, reg in self.agents.items()
            }
        }
    
    def list_all_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents with their details.
        
        Returns:
            List of agent information dictionaries
        """
        return [registration.to_dict() for registration in self.agents.values()]
    
    def _index_capabilities(self, agent_name: str, capabilities: List[str]):
        """
        Index agent capabilities for fast lookup.
        
        Args:
            agent_name: Agent identifier
            capabilities: List of capabilities to index
        """
        for capability in capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent_name)
    
    def _unindex_agent(self, agent_name: str):
        """
        Remove agent from capability index.
        
        Args:
            agent_name: Agent identifier to remove from index
        """
        if agent_name not in self.agents:
            return
        
        capabilities = self.agents[agent_name].capabilities
        for capability in capabilities:
            if capability in self.capability_index:
                self.capability_index[capability].discard(agent_name)
                # Clean up empty capability sets
                if not self.capability_index[capability]:
                    del self.capability_index[capability]


# Module-level convenience functions
_global_registry: Optional[AgentRegistry] = None


def get_global_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.
    
    Returns:
        Global AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(*args, **kwargs) -> bool:
    """
    Convenience function to register an agent in the global registry.
    """
    return get_global_registry().register_agent(*args, **kwargs)


def get_agents_by_capability(capability: str) -> List[str]:
    """
    Convenience function to get agents by capability from global registry.
    """
    return get_global_registry().get_agents_by_capability(capability)


if __name__ == "__main__":
    # Test the agent registry
    print("Agent Registry System Test")
    print("=" * 50)
    
    registry = AgentRegistry()
    
    # Register test agents
    class MockAgent:
        def __init__(self, name):
            self.name = name
    
    registry.register_agent(
        name="oracle",
        capabilities=["external_knowledge", "web_search", "mcp_discovery"],
        description="Gateway to external knowledge",
        agent_instance=MockAgent("oracle")
    )
    
    registry.register_agent(
        name="tool_creator",
        capabilities=["code_generation", "tool_creation", "text_synthesis"],
        description="Creates functional tools",
        agent_instance=MockAgent("tool_creator")
    )
    
    registry.register_agent(
        name="safety_agent",
        capabilities=["security_analysis", "code_validation"],
        description="Security and safety analysis",
        agent_instance=MockAgent("safety_agent")
    )
    
    # Test queries
    print("\nTest 1: Find agents with 'web_search' capability")
    print(f"Result: {registry.get_agents_by_capability('web_search')}")
    
    print("\nTest 2: Find agents with 'code_generation' capability")
    print(f"Result: {registry.get_agents_by_capability('code_generation')}")
    
    print("\nTest 3: Get all capabilities")
    for agent_name, caps in registry.get_all_capabilities().items():
        print(f"  {agent_name}: {caps}")
    
    print("\nTest 4: Find best agent for task")
    task = "What is the origin of Python programming language?"
    print(f"Task: {task}")
    print(f"Best agent: {registry.find_best_agent_for_task(task)}")
    
    print("\nTest 5: Registry statistics")
    stats = registry.get_registry_statistics()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Total capabilities: {stats['total_capabilities']}")
    print(f"Capability distribution: {stats['capability_distribution']}")
