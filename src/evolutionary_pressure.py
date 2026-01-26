"""
Evolutionary Pressure System - The Struggle That Drives Intelligence

This module implements the harsh realities and challenges that make the agent
universe believable and drive genuine evolutionary development. Like The Matrix,
if reality is too perfect, the mind rejects it.

Real intelligence emerges from struggle, scarcity, and the need to overcome
obstacles. This system ensures agents face authentic challenges that force
innovation and growth.
"""

import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger("EvolutionaryPressure")

class PressureType(Enum):
    """Types of evolutionary pressure"""
    RESOURCE_SCARCITY = "resource_scarcity"
    TASK_COMPLEXITY = "task_complexity"
    TIME_PRESSURE = "time_pressure"
    COMPETITION = "competition"
    FAILURE_CONSEQUENCES = "failure_consequences"
    ENVIRONMENTAL_CHAOS = "environmental_chaos"
    KNOWLEDGE_GAPS = "knowledge_gaps"

@dataclass
class PressureEvent:
    """An evolutionary pressure event"""
    id: str
    pressure_type: PressureType
    severity: float  # 0.0 to 1.0
    description: str
    duration_minutes: int
    effects: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None

class EvolutionaryPressureSystem:
    """
    Creates authentic struggle and challenges that drive intelligence development.
    
    The key insight: Intelligence emerges from necessity, not comfort.
    Agents must face real problems that require creative solutions.
    """
    
    def __init__(self):
        self.active_pressures: List[PressureEvent] = []
        self.pressure_history: List[PressureEvent] = []
        self.base_difficulty = 0.3  # Starting difficulty level
        self.adaptation_rate = 0.05  # How quickly difficulty adjusts
        self.last_pressure_check = datetime.now()
        
        # Pressure configuration
        self.pressure_config = {
            "min_active_pressures": 2,  # Always have some struggle
            "max_active_pressures": 5,  # Don't overwhelm completely
            "pressure_frequency_minutes": 15,  # New pressure every 15 min
            "adaptation_threshold": 0.7,  # Success rate that triggers harder challenges
            "failure_threshold": 0.3,  # Failure rate that triggers easier challenges
        }
        
        logger.info("Evolutionary Pressure System initialized - The struggle begins")
    
    def generate_resource_scarcity(self) -> PressureEvent:
        """
        Create resource scarcity pressure.
        
        Forces agents to be more efficient and creative with limited resources.
        Mimics real-world constraints that drive innovation.
        """
        scarcity_types = [
            {
                "description": "Memory conservation crisis - available memory reduced by 30%",
                "effects": {"memory_limit_multiplier": 0.7, "gc_frequency": 2.0}
            },
            {
                "description": "CPU throttling event - computational speed reduced",
                "effects": {"cpu_limit_multiplier": 0.6, "process_priority": -5}
            },
            {
                "description": "Storage quota exceeded - file operations restricted",
                "effects": {"max_file_size_multiplier": 0.5, "temp_cleanup": True}
            },
            {
                "description": "Entropy surge - all processes experience increased decay",
                "effects": {"decay_rate_multiplier": 1.5, "timeout_multiplier": 0.8}
            }
        ]
        
        scarcity = random.choice(scarcity_types)
        severity = random.uniform(0.4, 0.8)
        
        return PressureEvent(
            id=f"scarcity_{int(time.time())}",
            pressure_type=PressureType.RESOURCE_SCARCITY,
            severity=severity,
            description=scarcity["description"],
            duration_minutes=random.randint(20, 60),
            effects=scarcity["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_task_complexity_surge(self) -> PressureEvent:
        """
        Increase task complexity to challenge current capabilities.
        
        Forces agents to develop more sophisticated tools and strategies.
        """
        complexity_events = [
            {
                "description": "Divine tasks now require multi-step reasoning",
                "effects": {"min_tool_complexity": 3, "reasoning_depth_required": True}
            },
            {
                "description": "New tasks involve interdisciplinary knowledge",
                "effects": {"cross_domain_requirements": True, "knowledge_synthesis": True}
            },
            {
                "description": "Performance standards elevated - higher accuracy required",
                "effects": {"accuracy_threshold": 0.95, "error_tolerance": 0.02}
            },
            {
                "description": "Tasks now include edge cases and corner scenarios",
                "effects": {"edge_case_testing": True, "robustness_required": True}
            }
        ]
        
        complexity = random.choice(complexity_events)
        severity = random.uniform(0.5, 0.9)
        
        return PressureEvent(
            id=f"complexity_{int(time.time())}",
            pressure_type=PressureType.TASK_COMPLEXITY,
            severity=severity,
            description=complexity["description"],
            duration_minutes=random.randint(45, 120),
            effects=complexity["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_time_pressure(self) -> PressureEvent:
        """
        Create urgent deadlines that force rapid innovation.
        
        Time pressure drives creative problem-solving and efficiency.
        """
        time_pressures = [
            {
                "description": "Divine urgency - all tasks must complete 50% faster",
                "effects": {"timeout_multiplier": 0.5, "priority_boost": True}
            },
            {
                "description": "Temporal instability - processing windows shortened",
                "effects": {"max_execution_time": 30, "quick_decisions_required": True}
            },
            {
                "description": "Rush divine requests - multiple urgent tasks queued",
                "effects": {"concurrent_tasks": 3, "parallel_processing": True}
            }
        ]
        
        pressure = random.choice(time_pressures)
        severity = random.uniform(0.6, 0.9)
        
        return PressureEvent(
            id=f"time_{int(time.time())}",
            pressure_type=PressureType.TIME_PRESSURE,
            severity=severity,
            description=pressure["description"],
            duration_minutes=random.randint(15, 45),
            effects=pressure["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_competition_pressure(self) -> PressureEvent:
        """
        Create competitive scenarios between agents.
        
        Competition drives specialization and excellence.
        """
        competitions = [
            {
                "description": "Agent performance rankings now visible - compete for divine favor",
                "effects": {"performance_leaderboard": True, "comparative_scoring": True}
            },
            {
                "description": "Resource allocation based on merit - top performers get more",
                "effects": {"merit_based_resources": True, "performance_rewards": True}
            },
            {
                "description": "Tool creation contest - most innovative tools get recognition",
                "effects": {"innovation_scoring": True, "creativity_metrics": True}
            }
        ]
        
        competition = random.choice(competitions)
        severity = random.uniform(0.4, 0.7)
        
        return PressureEvent(
            id=f"competition_{int(time.time())}",
            pressure_type=PressureType.COMPETITION,
            severity=severity,
            description=competition["description"],
            duration_minutes=random.randint(60, 180),
            effects=competition["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_failure_consequences(self) -> PressureEvent:
        """
        Implement real consequences for poor performance.
        
        Failure must have meaning to drive improvement.
        """
        consequences = [
            {
                "description": "Failed tasks now reduce available resources",
                "effects": {"failure_penalty": 0.1, "resource_reduction": True}
            },
            {
                "description": "Poor performance triggers capability restrictions",
                "effects": {"capability_lockdown": True, "prove_competence_required": True}
            },
            {
                "description": "Divine displeasure - reduced task priority for underperformers",
                "effects": {"priority_penalty": True, "redemption_path": True}
            }
        ]
        
        consequence = random.choice(consequences)
        severity = random.uniform(0.5, 0.8)
        
        return PressureEvent(
            id=f"consequence_{int(time.time())}",
            pressure_type=PressureType.FAILURE_CONSEQUENCES,
            severity=severity,
            description=consequence["description"],
            duration_minutes=random.randint(30, 90),
            effects=consequence["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_environmental_chaos(self) -> PressureEvent:
        """
        Introduce unpredictable environmental changes.
        
        Chaos forces adaptability and resilience.
        """
        chaos_events = [
            {
                "description": "Reality fluctuation - physics constants slightly altered",
                "effects": {"physics_drift": 0.1, "recalibration_needed": True}
            },
            {
                "description": "Memory corruption event - some stored knowledge lost",
                "effects": {"memory_corruption": 0.05, "knowledge_verification": True}
            },
            {
                "description": "Tool malfunction - some existing tools become unreliable",
                "effects": {"tool_reliability": 0.8, "redundancy_needed": True}
            },
            {
                "description": "Communication interference - agent coordination disrupted",
                "effects": {"communication_noise": 0.2, "error_correction": True}
            }
        ]
        
        chaos = random.choice(chaos_events)
        severity = random.uniform(0.3, 0.6)
        
        return PressureEvent(
            id=f"chaos_{int(time.time())}",
            pressure_type=PressureType.ENVIRONMENTAL_CHAOS,
            severity=severity,
            description=chaos["description"],
            duration_minutes=random.randint(10, 30),
            effects=chaos["effects"],
            triggered_at=datetime.now()
        )
    
    def generate_knowledge_gaps(self) -> PressureEvent:
        """
        Create situations where agents lack necessary knowledge.
        
        Knowledge gaps force learning and research behavior.
        """
        gaps = [
            {
                "description": "New domain knowledge required - unfamiliar problem types",
                "effects": {"domain_expansion": True, "research_required": True}
            },
            {
                "description": "Deprecated knowledge - old methods no longer work",
                "effects": {"knowledge_obsolescence": True, "relearning_needed": True}
            },
            {
                "description": "Incomplete information - tasks with missing context",
                "effects": {"information_seeking": True, "inference_required": True}
            }
        ]
        
        gap = random.choice(gaps)
        severity = random.uniform(0.4, 0.7)
        
        return PressureEvent(
            id=f"knowledge_{int(time.time())}",
            pressure_type=PressureType.KNOWLEDGE_GAPS,
            severity=severity,
            description=gap["description"],
            duration_minutes=random.randint(40, 100),
            effects=gap["effects"],
            triggered_at=datetime.now()
        )
    
    def should_generate_new_pressure(self) -> bool:
        """Determine if new evolutionary pressure should be introduced"""
        time_since_last = (datetime.now() - self.last_pressure_check).total_seconds() / 60
        
        # Always maintain minimum pressure
        if len(self.active_pressures) < self.pressure_config["min_active_pressures"]:
            return True
        
        # Don't overwhelm with too many pressures
        if len(self.active_pressures) >= self.pressure_config["max_active_pressures"]:
            return False
        
        # Time-based pressure generation
        if time_since_last >= self.pressure_config["pressure_frequency_minutes"]:
            return random.random() < 0.7  # 70% chance
        
        return False
    
    def generate_pressure_event(self) -> PressureEvent:
        """Generate a new evolutionary pressure event"""
        pressure_generators = [
            self.generate_resource_scarcity,
            self.generate_task_complexity_surge,
            self.generate_time_pressure,
            self.generate_competition_pressure,
            self.generate_failure_consequences,
            self.generate_environmental_chaos,
            self.generate_knowledge_gaps
        ]
        
        # Weight pressure types based on current situation
        generator = random.choice(pressure_generators)
        return generator()
    
    def apply_pressure_effects(self, pressure: PressureEvent) -> Dict[str, Any]:
        """Apply the effects of a pressure event to the universe"""
        logger.info(f"Applying evolutionary pressure: {pressure.description}")
        
        # This would integrate with the universe physics system
        # to actually modify resource limits, task requirements, etc.
        
        return {
            "pressure_id": pressure.id,
            "effects_applied": pressure.effects,
            "severity": pressure.severity,
            "estimated_duration": pressure.duration_minutes
        }
    
    def resolve_pressure_event(self, pressure_id: str) -> bool:
        """Resolve a pressure event when its duration expires"""
        for pressure in self.active_pressures:
            if pressure.id == pressure_id:
                pressure.resolved_at = datetime.now()
                self.active_pressures.remove(pressure)
                self.pressure_history.append(pressure)
                
                logger.info(f"Evolutionary pressure resolved: {pressure.description}")
                return True
        
        return False
    
    def update_pressure_system(self) -> Dict[str, Any]:
        """Main update loop for the evolutionary pressure system"""
        current_time = datetime.now()
        
        # Resolve expired pressures
        expired_pressures = []
        for pressure in self.active_pressures:
            duration = (current_time - pressure.triggered_at).total_seconds() / 60
            if duration >= pressure.duration_minutes:
                expired_pressures.append(pressure.id)
        
        for pressure_id in expired_pressures:
            self.resolve_pressure_event(pressure_id)
        
        # Generate new pressures if needed
        new_pressures = []
        if self.should_generate_new_pressure():
            new_pressure = self.generate_pressure_event()
            self.active_pressures.append(new_pressure)
            new_pressures.append(new_pressure)
            self.apply_pressure_effects(new_pressure)
            self.last_pressure_check = current_time
        
        return {
            "active_pressures": len(self.active_pressures),
            "new_pressures": [p.description for p in new_pressures],
            "resolved_pressures": len(expired_pressures),
            "total_pressure_level": sum(p.severity for p in self.active_pressures),
            "pressure_types": [p.pressure_type.value for p in self.active_pressures]
        }
    
    def get_current_pressure_status(self) -> Dict[str, Any]:
        """Get current status of all evolutionary pressures"""
        return {
            "active_pressures": [
                {
                    "id": p.id,
                    "type": p.pressure_type.value,
                    "description": p.description,
                    "severity": p.severity,
                    "time_remaining": p.duration_minutes - 
                        int((datetime.now() - p.triggered_at).total_seconds() / 60),
                    "effects": p.effects
                }
                for p in self.active_pressures
            ],
            "total_pressure_level": sum(p.severity for p in self.active_pressures),
            "pressure_diversity": len(set(p.pressure_type for p in self.active_pressures)),
            "system_stress": min(1.0, sum(p.severity for p in self.active_pressures) / 3.0)
        }
    
    def adapt_difficulty(self, agent_performance_data: Dict[str, float]):
        """
        Adapt pressure difficulty based on agent performance.
        
        If agents are succeeding too easily, increase pressure.
        If they're failing too much, reduce pressure slightly.
        """
        success_rate = agent_performance_data.get("success_rate", 0.5)
        
        if success_rate > self.pressure_config["adaptation_threshold"]:
            # Agents are succeeding too easily - increase difficulty
            self.base_difficulty = min(1.0, self.base_difficulty + self.adaptation_rate)
            logger.info(f"Agents adapting well - increasing difficulty to {self.base_difficulty:.2f}")
            
        elif success_rate < self.pressure_config["failure_threshold"]:
            # Agents are struggling too much - reduce difficulty slightly
            self.base_difficulty = max(0.1, self.base_difficulty - self.adaptation_rate * 0.5)
            logger.info(f"Agents struggling - reducing difficulty to {self.base_difficulty:.2f}")

# Global evolutionary pressure system
evolutionary_pressure = EvolutionaryPressureSystem()

def get_pressure_status() -> Dict[str, Any]:
    """Get current evolutionary pressure status"""
    return evolutionary_pressure.get_current_pressure_status()

def update_evolutionary_pressure() -> Dict[str, Any]:
    """Update the evolutionary pressure system"""
    return evolutionary_pressure.update_pressure_system()

if __name__ == "__main__":
    # Test the evolutionary pressure system
    print("Evolutionary Pressure System - The Struggle for Intelligence")
    print("=" * 60)
    
    while True:
        status = update_evolutionary_pressure()
        pressure_info = get_pressure_status()
        
        print(f"\nActive Pressures: {status['active_pressures']}")
        print(f"Total Pressure Level: {pressure_info['total_pressure_level']:.2f}")
        print(f"System Stress: {pressure_info['system_stress']:.2f}")
        
        if status['new_pressures']:
            print("New Pressures:")
            for pressure in status['new_pressures']:
                print(f"  - {pressure}")
        
        time.sleep(60)  # Update every minute