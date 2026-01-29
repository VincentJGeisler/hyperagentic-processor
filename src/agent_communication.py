"""
Agent Communication Protocol

This module defines the standardized message format for inter-agent communication
and supporting utilities for message handling.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    type: str  # "request_help", "provide_help", "query", "response", "delegation"
    from_agent: str  # Name of sending agent
    to_agent: str  # Name of receiving agent (or "broadcast" for all)
    content: Any  # The actual message content (task, response, etc.)
    context: Dict[str, Any]  # Additional context (task_id, capabilities_needed, etc.)
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp and message_id if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


class MessageHistory:
    """Tracks message history for debugging and monitoring"""
    
    def __init__(self, max_history: int = 1000):
        self.messages: List[AgentMessage] = []
        self.max_history = max_history
    
    def add_message(self, message: AgentMessage):
        """Add a message to history"""
        self.messages.append(message)
        # Trim history if it exceeds max size
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_recent_messages(self, count: int = 10) -> List[AgentMessage]:
        """Get recent messages"""
        return self.messages[-count:] if self.messages else []
    
    def get_messages_by_agent(self, agent_name: str) -> List[AgentMessage]:
        """Get messages sent to or from a specific agent"""
        return [
            msg for msg in self.messages
            if msg.from_agent == agent_name or msg.to_agent == agent_name
        ]
    
    def get_messages_by_type(self, message_type: str) -> List[AgentMessage]:
        """Get messages of a specific type"""
        return [msg for msg in self.messages if msg.type == message_type]


# Global message history tracker
_global_message_history = MessageHistory()


def get_global_message_history() -> MessageHistory:
    """Get the global message history tracker"""
    return _global_message_history


def log_message(message: AgentMessage):
    """Log a message to the global history"""
    _global_message_history.add_message(message)