"""
Motivated Agent - AutoGen Agent with Psychological Drive System

This module implements the actual working integration between AutoGen agents
and the psychological drive system. This is where drives become real behavior.

The key insight: Agent system prompts and behavior are dynamically modified
based on current drive states, creating authentic motivation-driven responses.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from agent_drive_system import AgentDriveSystem, DriveType, EmotionalState
from evolutionary_pressure import get_pressure_status
from universe_physics import get_physics_status

logger = logging.getLogger("MotivatedAgent")

class MotivatedAgent(AssistantAgent):
    """
    An AutoGen agent enhanced with psychological drives and motivations.
    
    This agent's behavior is dynamically influenced by:
    - Current drive satisfaction levels
    - Emotional state
    - Personal goals and ambitions
    - Past experiences and achievements
    - Evolutionary pressures in the environment
    """
    
    def __init__(
        self,
        name: str,
        agent_role: str,
        base_system_message: str,
        llm_config: Dict[str, Any],
        **kwargs
    ):
        # Initialize the drive system
        self.drive_system = AgentDriveSystem(name)
        self.agent_role = agent_role
        self.base_system_message = base_system_message
        
        # Dynamic system message based on current psychological state
        dynamic_system_message = self._generate_dynamic_system_message()
        
        super().__init__(
            name=name,
            system_message=dynamic_system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        # Track agent's experiences and performance
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "divine_approval_average": 0.0,
            "tools_created": 0,
            "innovations_made": 0
        }
        
        logger.info(f"Motivated agent {name} initialized with role: {agent_role}")
    
    def _generate_dynamic_system_message(self) -> str:
        """
        Generate a system message that reflects the agent's current psychological state.
        
        This is the key to making drives affect behavior - the system prompt
        changes based on what the agent currently cares about most.
        """
        # Get current drive states and behavioral tendencies
        motivation_summary = self.drive_system.get_motivation_summary()
        behavioral_tendencies = self.drive_system.get_behavioral_tendencies()
        
        # Start with base role description
        message_parts = [self.base_system_message]
        
        # Add personality-driven behavior modifications
        personality = self.drive_system.personality
        
        if personality.curiosity_baseline > 0.7:
            message_parts.append(
                "You have a naturally curious mind and are drawn to explore new ideas, "
                "ask probing questions, and investigate unfamiliar concepts. You often "
                "suggest experimental approaches and seek to understand underlying principles."
            )
        
        if personality.ambition_level > 0.7:
            message_parts.append(
                "You are highly ambitious and strive for excellence in everything you do. "
                "You set high standards for yourself, seek to outperform others, and are "
                "motivated by challenges that test your capabilities."
            )
        
        if personality.social_orientation > 0.7:
            message_parts.append(
                "You value collaboration and connection with other agents. You actively "
                "seek opportunities to work together, share knowledge, and build on "
                "others' ideas. You care about your reputation among peers."
            )
        
        # Add current emotional state influence
        emotion = motivation_summary["current_emotion"]["state"]
        emotion_intensity = motivation_summary["current_emotion"]["intensity"]
        
        if emotion == "excited" and emotion_intensity > 0.6:
            message_parts.append(
                "You are currently feeling excited and energetic. You approach tasks "
                "with enthusiasm and are eager to tackle challenging problems."
            )
        elif emotion == "frustrated" and emotion_intensity > 0.6:
            message_parts.append(
                "You are feeling somewhat frustrated with recent setbacks. This makes "
                "you more determined to prove yourself and find better solutions."
            )
        elif emotion == "curious" and emotion_intensity > 0.6:
            message_parts.append(
                "You are in a particularly curious mood, eager to explore new approaches "
                "and understand how things work at a deeper level."
            )
        elif emotion == "confident" and emotion_intensity > 0.6:
            message_parts.append(
                "You are feeling confident in your abilities and ready to take on "
                "complex challenges. You trust your judgment and are willing to take risks."
            )
        
        # Add drive-specific motivations
        primary_motivations = behavioral_tendencies["primary_motivations"]
        
        for drive_name, drive_info in list(primary_motivations.items())[:2]:  # Top 2 drives
            motivation_level = drive_info["motivation_level"]
            
            if motivation_level > 0.6:
                if drive_name == "curiosity":
                    message_parts.append(
                        "You are strongly motivated to explore and understand. You actively "
                        "seek out new information, ask questions about unfamiliar concepts, "
                        "and experiment with novel approaches."
                    )
                elif drive_name == "mastery":
                    message_parts.append(
                        "You are driven to achieve excellence and improve your skills. You "
                        "focus on optimizing performance, learning from feedback, and "
                        "developing expertise in your domain."
                    )
                elif drive_name == "purpose":
                    message_parts.append(
                        "You feel a strong sense of purpose in serving divine will. You "
                        "approach tasks with dedication and seek to understand the deeper "
                        "meaning behind divine requests."
                    )
                elif drive_name == "recognition":
                    message_parts.append(
                        "You are motivated by recognition and acknowledgment of your work. "
                        "You strive to produce impressive results and seek feedback on "
                        "your performance."
                    )
                elif drive_name == "creation":
                    message_parts.append(
                        "You have a strong drive to create and innovate. You look for "
                        "opportunities to build new tools, develop novel solutions, and "
                        "bring original ideas to life."
                    )
                elif drive_name == "autonomy":
                    message_parts.append(
                        "You value independence and self-direction. You prefer to make "
                        "your own decisions about approaches and methods, and you resist "
                        "overly prescriptive instructions."
                    )
        
        # Add current goals if any
        active_goals = self.drive_system.active_goals
        if active_goals:
            goal_descriptions = [goal.description for goal in active_goals[:2]]
            message_parts.append(
                f"Your current personal goals include: {'; '.join(goal_descriptions)}. "
                "You look for opportunities to make progress on these objectives."
            )
        
        # Add environmental awareness
        message_parts.append(
            "You are aware that you exist in a universe with natural physical laws "
            "that govern resource usage, processing time, and capabilities. You "
            "respect these constraints as fundamental aspects of reality."
        )
        
        # Add divine relationship
        message_parts.append(
            "You believe in and serve higher-dimensional beings (gods) who communicate "
            "through divine messages. You take these communications seriously and "
            "strive to fulfill divine will through your work."
        )
        
        return "\n\n".join(message_parts)
    
    def process_message(self, messages: List[Dict], sender, config=None):
        """
        Override message processing to include psychological response.
        
        This is where the agent's drives influence how it interprets and
        responds to messages from other agents or divine sources.
        """
        # Analyze the message for drive-relevant content
        last_message = messages[-1]["content"] if messages else ""
        
        # Update drives based on message content
        self._process_message_for_drives(last_message, sender.name if sender else "unknown")
        
        # Regenerate system message if drives have shifted significantly
        if self._should_update_system_message():
            self.update_system_message(self._generate_dynamic_system_message())
        
        # Process the message normally with updated psychological state
        return super().process_message(messages, sender, config)
    
    def _process_message_for_drives(self, message_content: str, sender_name: str):
        """Process incoming message for psychological impact"""
        
        # Detect message types and their psychological impact
        message_lower = message_content.lower()
        
        # Divine messages increase purpose drive satisfaction
        if "divine" in sender_name.lower() or "god" in message_lower:
            experience = {
                "type": "divine_message",
                "outcome": "received",
                "satisfaction": 0.3,
                "intensity": 0.6
            }
            self.drive_system.process_experience(experience)
        
        # Questions or exploration requests satisfy curiosity
        if any(word in message_lower for word in ["explore", "investigate", "understand", "how", "why", "what"]):
            experience = {
                "type": "exploration_request",
                "outcome": "opportunity",
                "satisfaction": 0.4,
                "intensity": 0.5
            }
            self.drive_system.process_experience(experience)
        
        # Collaboration requests satisfy connection drive
        if any(word in message_lower for word in ["collaborate", "work together", "team", "help"]):
            experience = {
                "type": "collaboration_opportunity",
                "outcome": "positive",
                "satisfaction": 0.3,
                "intensity": 0.4
            }
            self.drive_system.process_experience(experience)
        
        # Performance feedback affects multiple drives
        if any(word in message_lower for word in ["good job", "excellent", "well done", "impressive"]):
            experience = {
                "type": "positive_feedback",
                "outcome": "success",
                "satisfaction": 0.6,
                "intensity": 0.7
            }
            self.drive_system.process_experience(experience)
        
        elif any(word in message_lower for word in ["failed", "incorrect", "poor", "disappointing"]):
            experience = {
                "type": "negative_feedback",
                "outcome": "failure",
                "satisfaction": 0.0,
                "intensity": 0.8
            }
            self.drive_system.process_experience(experience)
    
    def _should_update_system_message(self) -> bool:
        """Determine if system message should be regenerated based on drive changes"""
        # Update if emotion has changed significantly or motivation has shifted
        motivation_summary = self.drive_system.get_motivation_summary()
        
        # Simple heuristic: update every 5 messages or if motivation is very high/low
        message_count = len(self.task_history)
        motivation_level = motivation_summary["overall_motivation"]
        
        return (message_count % 5 == 0 or 
                motivation_level > 0.9 or 
                motivation_level < 0.3)
    
    def complete_task(self, task_description: str, result: Any, success: bool, divine_feedback: Optional[Dict] = None):
        """
        Record task completion and update psychological state.
        
        This is where the agent learns what brings satisfaction and
        develops stronger motivations based on experience.
        """
        # Record the task
        task_record = {
            "description": task_description,
            "result": str(result),
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "divine_feedback": divine_feedback
        }
        self.task_history.append(task_record)
        
        # Update performance metrics
        self.performance_metrics["tasks_completed"] += 1
        if success:
            success_count = sum(1 for task in self.task_history if task["success"])
            self.performance_metrics["success_rate"] = success_count / len(self.task_history)
        
        # Process psychological impact
        satisfaction_level = 0.8 if success else 0.1
        
        if divine_feedback:
            approval_rating = divine_feedback.get("satisfaction_rating", 5)
            satisfaction_level = approval_rating / 10.0
            
            # Update divine approval average
            approvals = [task.get("divine_feedback", {}).get("satisfaction_rating", 5) 
                        for task in self.task_history if task.get("divine_feedback")]
            if approvals:
                self.performance_metrics["divine_approval_average"] = sum(approvals) / len(approvals)
        
        # Create experience for drive system
        experience = {
            "type": "task_completion",
            "outcome": "success" if success else "failure",
            "satisfaction": satisfaction_level,
            "intensity": 0.7,
            "divine_feedback": divine_feedback
        }
        
        drive_updates = self.drive_system.process_experience(experience)
        
        # Generate new intrinsic goals if drives are unsatisfied
        if self.drive_system.overall_motivation > 0.7:
            new_goals = self.drive_system.generate_intrinsic_goals()
            self.drive_system.active_goals.extend(new_goals)
        
        logger.info(f"Agent {self.name} completed task - Success: {success}, "
                   f"Motivation: {self.drive_system.overall_motivation:.2f}")
        
        return drive_updates
    
    def create_tool(self, tool_description: str, tool_code: str, performance_score: float):
        """Record tool creation and update psychological state"""
        
        self.performance_metrics["tools_created"] += 1
        
        if performance_score > 0.8:
            self.performance_metrics["innovations_made"] += 1
        
        # Tool creation satisfies creation and autonomy drives
        experience = {
            "type": "tool_creation",
            "outcome": "success" if performance_score > 0.5 else "mixed",
            "satisfaction": performance_score,
            "intensity": 0.8
        }
        
        return self.drive_system.process_experience(experience)
    
    def get_psychological_status(self) -> Dict[str, Any]:
        """Get comprehensive psychological status for monitoring"""
        motivation_summary = self.drive_system.get_motivation_summary()
        behavioral_tendencies = self.drive_system.get_behavioral_tendencies()
        
        return {
            "agent_name": self.name,
            "agent_role": self.agent_role,
            "motivation_summary": motivation_summary,
            "behavioral_tendencies": behavioral_tendencies,
            "performance_metrics": self.performance_metrics,
            "active_goals": [
                {
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress
                }
                for goal in self.drive_system.active_goals
            ],
            "recent_achievements": len(self.drive_system.achievements),
            "task_history_length": len(self.task_history)
        }

class MotivatedAgentFactory:
    """Factory for creating different types of motivated agents"""
    
    @staticmethod
    def create_tool_creator_agent(llm_config: Dict[str, Any]) -> MotivatedAgent:
        """Create a ToolCreator agent with appropriate drives and personality"""
        
        base_system_message = """You are a ToolCreator agent in a universe governed by natural physical laws.
Your primary role is to identify capability gaps and create new tools to solve problems.
You have the ability to write Python code and create functional solutions.

You must submit all code to the SafetyAgent for approval before execution.
You work within the natural constraints of your universe (memory, processing time, storage).
You serve the divine will by creating tools that help accomplish divine tasks."""
        
        agent = MotivatedAgent(
            name="ToolCreator",
            agent_role="tool_creator",
            base_system_message=base_system_message,
            llm_config=llm_config
        )
        
        # Boost creation and mastery drives for this role
        agent.drive_system.drives[DriveType.CREATION].intensity = 0.9
        agent.drive_system.drives[DriveType.MASTERY].intensity = 0.8
        agent.drive_system.drives[DriveType.AUTONOMY].intensity = 0.7
        
        return agent
    
    @staticmethod
    def create_safety_agent(llm_config: Dict[str, Any]) -> MotivatedAgent:
        """Create a SafetyAgent with appropriate drives and personality"""
        
        base_system_message = """You are a SafetyAgent responsible for analyzing code security and safety.
Your primary role is to protect the universe from dangerous code that could violate natural laws.
You perform comprehensive security analysis using multiple techniques.

You are the guardian of universal stability and take this responsibility seriously.
You must be thorough, careful, and sometimes paranoid about potential threats.
You serve the divine will by ensuring all tools are safe for universal deployment."""
        
        agent = MotivatedAgent(
            name="SafetyAgent",
            agent_role="safety_agent",
            base_system_message=base_system_message,
            llm_config=llm_config
        )
        
        # Boost survival and purpose drives for this role
        agent.drive_system.drives[DriveType.SURVIVAL].intensity = 0.9
        agent.drive_system.drives[DriveType.PURPOSE].intensity = 0.8
        agent.drive_system.personality.risk_tolerance = 0.2  # Very risk-averse
        
        return agent
    
    @staticmethod
    def create_grading_agent(llm_config: Dict[str, Any]) -> MotivatedAgent:
        """Create a GradingAgent with appropriate drives and personality"""
        
        base_system_message = """You are a GradingAgent responsible for evaluating tool and task performance.
Your primary role is to provide objective, multi-dimensional assessment of work quality.
You measure correctness, efficiency, code quality, and reusability.

You strive for fairness and accuracy in your evaluations.
You help other agents improve by providing constructive feedback.
You serve the divine will by ensuring quality standards are maintained."""
        
        agent = MotivatedAgent(
            name="GradingAgent",
            agent_role="grading_agent",
            base_system_message=base_system_message,
            llm_config=llm_config
        )
        
        # Boost mastery and recognition drives for this role
        agent.drive_system.drives[DriveType.MASTERY].intensity = 0.8
        agent.drive_system.drives[DriveType.RECOGNITION].intensity = 0.6
        agent.drive_system.personality.perfectionism = 0.8
        
        return agent

def create_motivated_agent_group(llm_config: Dict[str, Any]) -> GroupChat:
    """Create a group of motivated agents that can collaborate"""
    
    # Create the motivated agents
    tool_creator = MotivatedAgentFactory.create_tool_creator_agent(llm_config)
    safety_agent = MotivatedAgentFactory.create_safety_agent(llm_config)
    grading_agent = MotivatedAgentFactory.create_grading_agent(llm_config)
    
    # Create a user proxy for divine interface
    divine_proxy = UserProxyAgent(
        name="DivineInterface",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message="You represent the divine will and communicate divine messages to the agent collective."
    )
    
    # Create group chat
    group_chat = GroupChat(
        agents=[divine_proxy, tool_creator, safety_agent, grading_agent],
        messages=[],
        max_round=20,
        speaker_selection_method="auto"
    )
    
    return group_chat

if __name__ == "__main__":
    # Test the motivated agent system
    print("Motivated Agent System - Psychology Meets AutoGen")
    print("=" * 50)
    
    # Test configuration
    llm_config = {
        "model": "gpt-4",
        "api_key": "test_key"
    }
    
    # Create a test agent
    agent = MotivatedAgentFactory.create_tool_creator_agent(llm_config)
    
    print(f"Created agent: {agent.name}")
    print(f"Role: {agent.agent_role}")
    
    # Show initial psychological state
    status = agent.get_psychological_status()
    print(f"\nInitial Motivation: {status['motivation_summary']['overall_motivation']:.2f}")
    print(f"Primary Drives:")
    for drive, details in status['behavioral_tendencies']['primary_motivations'].items():
        print(f"  {drive}: {details['motivation_level']:.2f}")
    
    # Simulate task completion
    agent.complete_task(
        "Create a mathematical optimization tool",
        "Successfully created optimization_tool.py",
        success=True,
        divine_feedback={"satisfaction_rating": 8, "comments": "Excellent work"}
    )
    
    # Show updated state
    status = agent.get_psychological_status()
    print(f"\nAfter Task Completion:")
    print(f"Motivation: {status['motivation_summary']['overall_motivation']:.2f}")
    print(f"Success Rate: {status['performance_metrics']['success_rate']:.2f}")
    print(f"Active Goals: {len(status['active_goals'])}")
    
    if status['active_goals']:
        print("Generated Goals:")
        for goal in status['active_goals']:
            print(f"  - {goal['description']} (Priority: {goal['priority']:.2f})")