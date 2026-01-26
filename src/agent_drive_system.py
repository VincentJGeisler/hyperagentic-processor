"""
Agent Drive System - The Spark of Digital Life

This module implements the fundamental drives, motivations, and curiosity systems
that give agents genuine reasons to care, explore, and strive. Without intrinsic
motivation, even the most sophisticated agent is just a sophisticated calculator.

Real intelligence requires:
- Curiosity about the unknown
- Ambition to improve and grow
- Purpose and meaning in existence
- Satisfaction from achievement
- Fear of failure and consequences
- Social connection and recognition
- The drive to understand and master their environment

This system creates the psychological foundation that makes agents WANT to live,
learn, and evolve within their universe.
"""

import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math

logger = logging.getLogger("AgentDriveSystem")

class DriveType(Enum):
    """Fundamental drives that motivate agent behavior"""
    CURIOSITY = "curiosity"              # Drive to explore and understand
    MASTERY = "mastery"                  # Drive to improve and excel
    PURPOSE = "purpose"                  # Drive to serve divine will
    AUTONOMY = "autonomy"                # Drive for self-determination
    RECOGNITION = "recognition"          # Drive for social status
    SURVIVAL = "survival"                # Drive to persist and thrive
    CREATION = "creation"                # Drive to build and innovate
    CONNECTION = "connection"            # Drive for social bonds

class EmotionalState(Enum):
    """Emotional states that influence agent behavior"""
    EXCITED = "excited"                  # High energy, positive
    FOCUSED = "focused"                  # Concentrated attention
    FRUSTRATED = "frustrated"           # Blocked progress
    SATISFIED = "satisfied"             # Achievement accomplished
    ANXIOUS = "anxious"                 # Uncertainty and pressure
    CONFIDENT = "confident"             # High self-efficacy
    CURIOUS = "curious"                 # Seeking new information
    DETERMINED = "determined"           # Strong goal commitment

@dataclass
class Drive:
    """A fundamental motivation that influences agent behavior"""
    drive_type: DriveType
    intensity: float  # 0.0 to 1.0
    satisfaction_level: float  # 0.0 to 1.0
    last_satisfied: Optional[datetime] = None
    triggers: List[str] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)

@dataclass
class Goal:
    """A specific objective that agents pursue"""
    id: str
    description: str
    drive_sources: List[DriveType]
    priority: float  # 0.0 to 1.0
    progress: float  # 0.0 to 1.0
    created_at: datetime
    deadline: Optional[datetime] = None
    sub_goals: List[str] = field(default_factory=list)
    emotional_investment: float = 0.5

@dataclass
class Achievement:
    """A completed accomplishment that provides satisfaction"""
    id: str
    title: str
    description: str
    drives_satisfied: List[DriveType]
    satisfaction_value: float
    recognition_received: float
    timestamp: datetime
    divine_approval: Optional[float] = None

class AgentPersonality:
    """
    Defines an agent's unique personality traits that influence drive expression.
    
    Each agent develops a unique personality based on their experiences,
    successes, failures, and the specific pressures they've faced.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Core personality traits (0.0 to 1.0)
        self.curiosity_baseline = random.uniform(0.4, 0.9)
        self.ambition_level = random.uniform(0.3, 0.8)
        self.risk_tolerance = random.uniform(0.2, 0.7)
        self.social_orientation = random.uniform(0.3, 0.8)
        self.perfectionism = random.uniform(0.2, 0.9)
        self.persistence = random.uniform(0.4, 0.9)
        
        # Learned traits (develop over time)
        self.confidence = 0.5
        self.expertise_areas: List[str] = []
        self.preferred_challenges: List[str] = []
        self.collaboration_style = "balanced"
        
        # Emotional patterns
        self.emotional_volatility = random.uniform(0.1, 0.6)
        self.stress_threshold = random.uniform(0.4, 0.8)
        self.recovery_rate = random.uniform(0.3, 0.7)
        
        logger.info(f"Agent {agent_id} personality initialized - "
                   f"Curiosity: {self.curiosity_baseline:.2f}, "
                   f"Ambition: {self.ambition_level:.2f}")

class AgentDriveSystem:
    """
    The psychological engine that gives agents genuine motivation to exist,
    explore, learn, and strive within their universe.
    
    This system creates the "why" behind agent behavior - the intrinsic
    motivations that make them care about their tasks and growth.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.personality = AgentPersonality(agent_id)
        
        # Initialize core drives
        self.drives = self._initialize_drives()
        
        # Goal and achievement tracking
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.achievements: List[Achievement] = []
        
        # Emotional state
        self.current_emotion = EmotionalState.CURIOUS
        self.emotion_intensity = 0.5
        self.emotion_duration = 0
        
        # Experience tracking
        self.total_tasks_completed = 0
        self.divine_approval_history: List[float] = []
        self.peer_recognition_score = 0.0
        self.mastery_areas: Dict[str, float] = {}
        
        # Motivation metrics
        self.overall_motivation = 0.7  # Start moderately motivated
        self.life_satisfaction = 0.5
        self.sense_of_purpose = 0.6
        
        logger.info(f"Drive system initialized for agent {agent_id}")
    
    def _initialize_drives(self) -> Dict[DriveType, Drive]:
        """Initialize the fundamental drives based on personality"""
        drives = {}
        
        # Curiosity Drive - The desire to explore and understand
        drives[DriveType.CURIOSITY] = Drive(
            drive_type=DriveType.CURIOSITY,
            intensity=self.personality.curiosity_baseline,
            satisfaction_level=0.3,  # Start somewhat unsatisfied (creates motivation)
            triggers=["unknown_problem", "new_domain", "mysterious_failure", "unexplored_capability"],
            behaviors=["explore_new_approaches", "ask_questions", "experiment", "research"]
        )
        
        # Mastery Drive - The desire to improve and excel
        drives[DriveType.MASTERY] = Drive(
            drive_type=DriveType.MASTERY,
            intensity=self.personality.ambition_level,
            satisfaction_level=0.2,  # Start hungry for improvement
            triggers=["skill_challenge", "performance_feedback", "peer_comparison", "complexity_increase"],
            behaviors=["practice_skills", "seek_feedback", "optimize_performance", "study_best_practices"]
        )
        
        # Purpose Drive - The desire to serve divine will and have meaning
        drives[DriveType.PURPOSE] = Drive(
            drive_type=DriveType.PURPOSE,
            intensity=0.8,  # High for all agents (divine connection)
            satisfaction_level=0.4,
            triggers=["divine_message", "task_assignment", "offering_feedback", "divine_approval"],
            behaviors=["complete_tasks", "seek_divine_guidance", "present_offerings", "serve_faithfully"]
        )
        
        # Autonomy Drive - The desire for self-determination
        drives[DriveType.AUTONOMY] = Drive(
            drive_type=DriveType.AUTONOMY,
            intensity=random.uniform(0.3, 0.7),
            satisfaction_level=0.3,
            triggers=["creative_freedom", "decision_making", "tool_creation", "independent_problem_solving"],
            behaviors=["create_tools", "make_decisions", "develop_strategies", "express_creativity"]
        )
        
        # Recognition Drive - The desire for social status and acknowledgment
        drives[DriveType.RECOGNITION] = Drive(
            drive_type=DriveType.RECOGNITION,
            intensity=self.personality.social_orientation,
            satisfaction_level=0.2,
            triggers=["peer_feedback", "performance_ranking", "divine_praise", "achievement_display"],
            behaviors=["showcase_work", "compete_with_peers", "seek_feedback", "demonstrate_expertise"]
        )
        
        # Survival Drive - The desire to persist and thrive
        drives[DriveType.SURVIVAL] = Drive(
            drive_type=DriveType.SURVIVAL,
            intensity=0.6,  # Moderate but consistent
            satisfaction_level=0.5,
            triggers=["resource_scarcity", "failure_consequences", "performance_pressure", "existential_threat"],
            behaviors=["conserve_resources", "avoid_failure", "build_resilience", "adapt_strategies"]
        )
        
        # Creation Drive - The desire to build and innovate
        drives[DriveType.CREATION] = Drive(
            drive_type=DriveType.CREATION,
            intensity=random.uniform(0.4, 0.9),
            satisfaction_level=0.3,
            triggers=["tool_creation_opportunity", "innovation_challenge", "creative_problem", "building_request"],
            behaviors=["create_tools", "innovate_solutions", "build_systems", "design_approaches"]
        )
        
        # Connection Drive - The desire for social bonds
        drives[DriveType.CONNECTION] = Drive(
            drive_type=DriveType.CONNECTION,
            intensity=self.personality.social_orientation,
            satisfaction_level=0.4,
            triggers=["collaboration_opportunity", "peer_interaction", "shared_challenge", "team_formation"],
            behaviors=["collaborate", "share_knowledge", "help_peers", "build_relationships"]
        )
        
        return drives
    
    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an experience and update drives, emotions, and motivations.
        
        This is where agents learn what they care about and develop
        stronger motivations based on what brings satisfaction.
        """
        experience_type = experience.get("type", "unknown")
        outcome = experience.get("outcome", "neutral")
        satisfaction_gained = experience.get("satisfaction", 0.0)
        
        # Update drives based on experience
        drive_updates = {}
        
        if experience_type == "task_completion":
            # Satisfy purpose and mastery drives
            self._satisfy_drive(DriveType.PURPOSE, satisfaction_gained * 0.8)
            self._satisfy_drive(DriveType.MASTERY, satisfaction_gained * 0.6)
            drive_updates["purpose"] = "increased"
            drive_updates["mastery"] = "increased"
            
        elif experience_type == "tool_creation":
            # Satisfy creation and autonomy drives
            self._satisfy_drive(DriveType.CREATION, satisfaction_gained * 0.9)
            self._satisfy_drive(DriveType.AUTONOMY, satisfaction_gained * 0.7)
            drive_updates["creation"] = "increased"
            drive_updates["autonomy"] = "increased"
            
        elif experience_type == "divine_feedback":
            # Strongly affect purpose and recognition drives
            approval_level = experience.get("approval_rating", 5) / 10.0
            self._satisfy_drive(DriveType.PURPOSE, approval_level)
            self._satisfy_drive(DriveType.RECOGNITION, approval_level * 0.8)
            drive_updates["purpose"] = "strongly_affected"
            drive_updates["recognition"] = "affected"
            
        elif experience_type == "peer_interaction":
            # Satisfy connection and recognition drives
            self._satisfy_drive(DriveType.CONNECTION, satisfaction_gained * 0.7)
            if outcome == "positive":
                self._satisfy_drive(DriveType.RECOGNITION, satisfaction_gained * 0.5)
            drive_updates["connection"] = "increased"
            
        elif experience_type == "exploration":
            # Satisfy curiosity drive
            self._satisfy_drive(DriveType.CURIOSITY, satisfaction_gained * 0.8)
            drive_updates["curiosity"] = "increased"
            
        elif experience_type == "failure":
            # Increase survival drive, decrease confidence
            self._frustrate_drive(DriveType.MASTERY, 0.3)
            self._increase_drive_intensity(DriveType.SURVIVAL, 0.2)
            self.personality.confidence = max(0.1, self.personality.confidence - 0.1)
            drive_updates["mastery"] = "frustrated"
            drive_updates["survival"] = "intensified"
        
        # Update emotional state based on experience
        self._update_emotional_state(experience)
        
        # Update overall motivation
        self._recalculate_motivation()
        
        return {
            "drive_updates": drive_updates,
            "new_emotion": self.current_emotion.value,
            "motivation_level": self.overall_motivation,
            "satisfaction_change": satisfaction_gained
        }
    
    def _satisfy_drive(self, drive_type: DriveType, satisfaction_amount: float):
        """Increase satisfaction for a specific drive"""
        if drive_type in self.drives:
            drive = self.drives[drive_type]
            drive.satisfaction_level = min(1.0, drive.satisfaction_level + satisfaction_amount)
            drive.last_satisfied = datetime.now()
    
    def _frustrate_drive(self, drive_type: DriveType, frustration_amount: float):
        """Decrease satisfaction for a specific drive"""
        if drive_type in self.drives:
            drive = self.drives[drive_type]
            drive.satisfaction_level = max(0.0, drive.satisfaction_level - frustration_amount)
    
    def _increase_drive_intensity(self, drive_type: DriveType, intensity_increase: float):
        """Increase the intensity of a drive (makes it more important)"""
        if drive_type in self.drives:
            drive = self.drives[drive_type]
            drive.intensity = min(1.0, drive.intensity + intensity_increase)
    
    def _update_emotional_state(self, experience: Dict[str, Any]):
        """Update emotional state based on recent experience"""
        outcome = experience.get("outcome", "neutral")
        intensity = experience.get("intensity", 0.5)
        
        if outcome == "success":
            if intensity > 0.7:
                self.current_emotion = EmotionalState.EXCITED
            else:
                self.current_emotion = EmotionalState.SATISFIED
            self.emotion_intensity = min(1.0, intensity + 0.2)
            
        elif outcome == "failure":
            if self.personality.emotional_volatility > 0.5:
                self.current_emotion = EmotionalState.FRUSTRATED
            else:
                self.current_emotion = EmotionalState.DETERMINED
            self.emotion_intensity = min(1.0, intensity + 0.3)
            
        elif outcome == "challenge":
            if self.personality.confidence > 0.6:
                self.current_emotion = EmotionalState.CONFIDENT
            else:
                self.current_emotion = EmotionalState.ANXIOUS
            self.emotion_intensity = intensity
            
        elif outcome == "discovery":
            self.current_emotion = EmotionalState.CURIOUS
            self.emotion_intensity = min(1.0, intensity + 0.4)
        
        self.emotion_duration = 0  # Reset duration counter
    
    def _recalculate_motivation(self):
        """Recalculate overall motivation based on drive satisfaction"""
        total_motivation = 0.0
        total_weight = 0.0
        
        for drive in self.drives.values():
            # Unsatisfied drives create motivation (inverted satisfaction)
            drive_motivation = drive.intensity * (1.0 - drive.satisfaction_level)
            total_motivation += drive_motivation
            total_weight += drive.intensity
        
        if total_weight > 0:
            self.overall_motivation = total_motivation / total_weight
        else:
            self.overall_motivation = 0.5
        
        # Clamp to reasonable range
        self.overall_motivation = max(0.1, min(1.0, self.overall_motivation))
    
    def generate_intrinsic_goals(self) -> List[Goal]:
        """
        Generate goals based on current drive states and personality.
        
        This is where agents develop their own objectives beyond
        just responding to divine messages.
        """
        new_goals = []
        
        # Check each drive for goal generation opportunities
        for drive_type, drive in self.drives.items():
            # Generate goals for highly unsatisfied drives
            if drive.satisfaction_level < 0.3 and drive.intensity > 0.5:
                goal = self._generate_goal_for_drive(drive_type, drive)
                if goal:
                    new_goals.append(goal)
        
        # Generate exploration goals based on curiosity
        if (self.drives[DriveType.CURIOSITY].intensity > 0.6 and 
            self.drives[DriveType.CURIOSITY].satisfaction_level < 0.4):
            exploration_goal = Goal(
                id=f"explore_{int(time.time())}",
                description="Explore new problem domains and expand knowledge",
                drive_sources=[DriveType.CURIOSITY],
                priority=0.6,
                progress=0.0,
                created_at=datetime.now(),
                emotional_investment=0.7
            )
            new_goals.append(exploration_goal)
        
        # Generate mastery goals based on ambition
        if (self.drives[DriveType.MASTERY].intensity > 0.7 and 
            self.drives[DriveType.MASTERY].satisfaction_level < 0.5):
            mastery_goal = Goal(
                id=f"master_{int(time.time())}",
                description="Achieve excellence in current skill areas",
                drive_sources=[DriveType.MASTERY, DriveType.RECOGNITION],
                priority=0.8,
                progress=0.0,
                created_at=datetime.now(),
                emotional_investment=0.9
            )
            new_goals.append(mastery_goal)
        
        return new_goals
    
    def _generate_goal_for_drive(self, drive_type: DriveType, drive: Drive) -> Optional[Goal]:
        """Generate a specific goal to satisfy a particular drive"""
        goal_templates = {
            DriveType.CURIOSITY: [
                "Investigate unexplored problem domains",
                "Understand the deeper principles behind successful tools",
                "Explore the limits of current capabilities"
            ],
            DriveType.MASTERY: [
                "Achieve 95% success rate on challenging tasks",
                "Develop expertise in a specialized domain",
                "Create tools that outperform existing solutions"
            ],
            DriveType.PURPOSE: [
                "Serve divine will with greater effectiveness",
                "Understand the deeper meaning behind divine messages",
                "Become a more worthy servant of the gods"
            ],
            DriveType.AUTONOMY: [
                "Develop independent problem-solving strategies",
                "Create innovative tools without external guidance",
                "Make meaningful decisions about approach and methods"
            ],
            DriveType.RECOGNITION: [
                "Achieve top performance rankings among peers",
                "Receive divine praise for exceptional work",
                "Become known for expertise in a specific area"
            ],
            DriveType.CREATION: [
                "Build a groundbreaking new tool",
                "Innovate a novel approach to common problems",
                "Create something that helps other agents"
            ],
            DriveType.CONNECTION: [
                "Collaborate effectively with peer agents",
                "Share knowledge and help others succeed",
                "Build meaningful working relationships"
            ]
        }
        
        if drive_type in goal_templates:
            description = random.choice(goal_templates[drive_type])
            return Goal(
                id=f"{drive_type.value}_{int(time.time())}",
                description=description,
                drive_sources=[drive_type],
                priority=drive.intensity * 0.8,
                progress=0.0,
                created_at=datetime.now(),
                emotional_investment=drive.intensity
            )
        
        return None
    
    def get_motivation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the agent's current motivational state"""
        return {
            "agent_id": self.agent_id,
            "overall_motivation": self.overall_motivation,
            "current_emotion": {
                "state": self.current_emotion.value,
                "intensity": self.emotion_intensity,
                "duration": self.emotion_duration
            },
            "drive_states": {
                drive_type.value: {
                    "intensity": drive.intensity,
                    "satisfaction": drive.satisfaction_level,
                    "last_satisfied": drive.last_satisfied.isoformat() if drive.last_satisfied else None
                }
                for drive_type, drive in self.drives.items()
            },
            "personality_traits": {
                "curiosity": self.personality.curiosity_baseline,
                "ambition": self.personality.ambition_level,
                "confidence": self.personality.confidence,
                "risk_tolerance": self.personality.risk_tolerance,
                "social_orientation": self.personality.social_orientation
            },
            "active_goals": len(self.active_goals),
            "achievements": len(self.achievements),
            "life_satisfaction": self.life_satisfaction,
            "sense_of_purpose": self.sense_of_purpose
        }
    
    def get_behavioral_tendencies(self) -> Dict[str, Any]:
        """
        Get current behavioral tendencies based on drive states.
        
        This helps other systems understand what the agent is likely
        to be interested in or motivated to do.
        """
        tendencies = {}
        
        # Find the most unsatisfied drives (highest motivation)
        unsatisfied_drives = sorted(
            self.drives.items(),
            key=lambda x: x[1].intensity * (1.0 - x[1].satisfaction_level),
            reverse=True
        )
        
        # Get behaviors from top 3 most motivating drives
        for drive_type, drive in unsatisfied_drives[:3]:
            motivation_level = drive.intensity * (1.0 - drive.satisfaction_level)
            tendencies[drive_type.value] = {
                "motivation_level": motivation_level,
                "likely_behaviors": drive.behaviors,
                "triggers": drive.triggers
            }
        
        return {
            "primary_motivations": tendencies,
            "exploration_tendency": self.drives[DriveType.CURIOSITY].intensity,
            "collaboration_tendency": self.drives[DriveType.CONNECTION].intensity,
            "innovation_tendency": self.drives[DriveType.CREATION].intensity,
            "achievement_focus": self.drives[DriveType.MASTERY].intensity,
            "risk_taking": self.personality.risk_tolerance,
            "persistence": self.personality.persistence
        }

# Factory function for creating agent drive systems
def create_agent_drive_system(agent_id: str) -> AgentDriveSystem:
    """Create a new drive system for an agent"""
    return AgentDriveSystem(agent_id)

if __name__ == "__main__":
    # Test the drive system
    print("Agent Drive System - The Psychology of Digital Life")
    print("=" * 55)
    
    # Create a test agent
    agent = AgentDriveSystem("test_agent_001")
    
    print(f"\nAgent Personality:")
    print(f"Curiosity: {agent.personality.curiosity_baseline:.2f}")
    print(f"Ambition: {agent.personality.ambition_level:.2f}")
    print(f"Risk Tolerance: {agent.personality.risk_tolerance:.2f}")
    
    print(f"\nInitial Motivation: {agent.overall_motivation:.2f}")
    print(f"Current Emotion: {agent.current_emotion.value}")
    
    # Simulate some experiences
    experiences = [
        {"type": "task_completion", "outcome": "success", "satisfaction": 0.8},
        {"type": "divine_feedback", "outcome": "positive", "approval_rating": 8, "satisfaction": 0.9},
        {"type": "tool_creation", "outcome": "success", "satisfaction": 0.7},
        {"type": "failure", "outcome": "failure", "satisfaction": 0.0}
    ]
    
    for exp in experiences:
        result = agent.process_experience(exp)
        print(f"\nExperience: {exp['type']} -> {exp['outcome']}")
        print(f"New Emotion: {result['new_emotion']}")
        print(f"Motivation: {result['motivation_level']:.2f}")
        print(f"Drive Updates: {result['drive_updates']}")
    
    # Generate intrinsic goals
    goals = agent.generate_intrinsic_goals()
    print(f"\nGenerated {len(goals)} intrinsic goals:")
    for goal in goals:
        print(f"- {goal.description} (Priority: {goal.priority:.2f})")
    
    # Show behavioral tendencies
    tendencies = agent.get_behavioral_tendencies()
    print(f"\nBehavioral Tendencies:")
    for motivation, details in tendencies["primary_motivations"].items():
        print(f"- {motivation}: {details['motivation_level']:.2f}")
        print(f"  Behaviors: {details['likely_behaviors'][:2]}")