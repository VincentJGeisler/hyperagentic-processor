"""
Hyperagentic Processor - Main Orchestration System

This is the main entry point that brings together all components:
- Motivated AutoGen agents with psychological drives
- Divine interface for task communication
- Evolutionary pressure system
- Universe physics enforcement
- Real-time monitoring and adaptation

This creates the complete organic AGI development environment.
"""

import asyncio
import logging
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager

# Import our core systems
from divine_interface import DivineInterface, app as divine_app
from motivated_agent import MotivatedAgentFactory, create_motivated_agent_group
from tool_creator_agent import create_tool_creator_agent
from safety_agent import create_safety_agent
from grading_agent import create_grading_agent
from evolutionary_pressure import evolutionary_pressure, update_evolutionary_pressure
from universe_physics import universe_physics, check_natural_laws
from agent_drive_system import DriveType

# Import LLM configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from llm_config import get_llm_config, validate_llm_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HyperagenticProcessor")

class HyperagenticOrchestrator:
    """
    Main orchestrator that manages the complete agent universe.
    
    This system coordinates:
    - Agent psychological states and motivations
    - Task assignment and completion tracking
    - Evolutionary pressure adaptation
    - Universe physics monitoring
    - Divine communication and feedback
    """
    
    def __init__(self):
        self.universe_id = "reality_001"
        self.start_time = datetime.now()
        
        # Initialize core systems
        self.divine_interface = DivineInterface()
        
        # LLM Configuration - Using Groq like the original class example
        self.llm_config = get_llm_config()
        
        # Validate configuration
        if not validate_llm_config():
            logger.warning("LLM configuration validation failed - using test configuration")
            from llm_config import TEST_LLM_CONFIG
            self.llm_config = TEST_LLM_CONFIG
        
        # Create motivated agents
        self.agents = self._initialize_agents()
        
        # Create agent group for collaboration
        self.agent_group = create_motivated_agent_group(self.llm_config)
        
        # System state tracking
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: List[Dict] = []
        self.system_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "tools_created": 0,
            "agent_satisfaction": 0.0,
            "universe_stability": 1.0
        }
        
        logger.info(f"Hyperagentic Processor initialized - Universe {self.universe_id}")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize the motivated agent collective"""
        agents = {}
        
        # Create specialized agents with functional capabilities
        agents["tool_creator"] = create_tool_creator_agent(self.llm_config)
        agents["safety_agent"] = create_safety_agent(self.llm_config)
        agents["grading_agent"] = create_grading_agent(self.llm_config)
        
        # Create Oracle agent with reference to SafetyAgent
        from oracle_agent import create_oracle_agent
        agents["oracle"] = create_oracle_agent(self.llm_config, agents["safety_agent"])
        
        # TODO: Add more agent types as we implement them
        # agents["memory_keeper"] = MotivatedAgentFactory.create_memory_keeper_agent(self.llm_config)
        # agents["reflection_agent"] = MotivatedAgentFactory.create_reflection_agent(self.llm_config)
        # agents["altar_agent"] = MotivatedAgentFactory.create_altar_agent(self.llm_config)
        
        logger.info(f"Initialized {len(agents)} motivated agents with functional capabilities")
        return agents
    
    async def process_divine_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a divine message and coordinate agent response.
        
        This is where divine tasks get translated into agent motivation
        and collaborative problem-solving.
        """
        task_id = message.get("message_id")
        divine_text = message.get("message")
        priority = message.get("priority", 5)
        
        logger.info(f"Processing divine message: {task_id}")
        
        # Create task record
        task_record = {
            "id": task_id,
            "divine_message": divine_text,
            "priority": priority,
            "assigned_agents": [],
            "status": "in_progress",
            "start_time": datetime.now(),
            "agent_responses": [],
            "tools_created": [],
            "final_result": None
        }
        
        self.active_tasks[task_id] = task_record
        
        # Analyze task to determine which agents should be involved
        involved_agents = self._analyze_task_requirements(divine_text)
        task_record["assigned_agents"] = involved_agents
        
        # Notify agents of the divine message (affects their psychology)
        for agent_name in involved_agents:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                
                # Process the divine message psychologically
                agent._process_message_for_drives(divine_text, "DivineInterface")
                
                # Update agent's system message based on new motivation
                if agent._should_update_system_message():
                    agent.update_system_message(agent._generate_dynamic_system_message())
        
        # Coordinate agent collaboration to address the task
        result = await self._coordinate_agent_collaboration(task_record)
        
        # Complete the task and update metrics
        await self._complete_task(task_record, result)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "agents_involved": involved_agents,
            "completion_time": datetime.now().isoformat()
        }
    
    def _analyze_task_requirements(self, divine_text: str) -> List[str]:
        """Analyze divine message to determine which agents should be involved"""
        text_lower = divine_text.lower()
        involved_agents = []
        
        # Always involve tool creator for most tasks
        if any(word in text_lower for word in ["create", "build", "develop", "tool", "solution"]):
            involved_agents.append("tool_creator")
        
        # Involve safety agent if code/security is mentioned
        if any(word in text_lower for word in ["safe", "secure", "analyze", "check", "validate"]):
            involved_agents.append("safety_agent")
        
        # Involve grading agent for evaluation tasks
        if any(word in text_lower for word in ["evaluate", "assess", "grade", "measure", "performance"]):
            involved_agents.append("grading_agent")
        
        # Default to tool creator if no specific requirements detected
        if not involved_agents:
            involved_agents.append("tool_creator")
        
        return involved_agents
    
    async def _coordinate_agent_collaboration(self, task_record: Dict) -> Dict[str, Any]:
        """
        Coordinate multiple agents to collaboratively address a divine task.
        
        This is where the magic happens - agents with different drives and
        personalities work together, each contributing their unique perspective.
        """
        task_id = task_record["id"]
        divine_message = task_record["divine_message"]
        involved_agents = task_record["assigned_agents"]
        
        logger.info(f"Coordinating collaboration for task {task_id} with agents: {involved_agents}")
        
        collaboration_result = {
            "approach_decided": None,
            "tools_created": [],
            "safety_analysis": None,
            "performance_evaluation": None,
            "agent_contributions": {}
        }
        
        # Tool Creator's contribution - actually create a functional tool
        if "tool_creator" in involved_agents:
            tool_creator = self.agents["tool_creator"]
            
            # Use the actual tool creation functionality
            tool_creation_result = tool_creator.create_tool_from_requirements(divine_message)
            
            if tool_creation_result["success"]:
                collaboration_result["tools_created"].append({
                    "tool_id": tool_creation_result["tool_id"],
                    "tool_name": tool_creation_result["tool_name"],
                    "tool_code": tool_creation_result["code"],
                    "creation_time": tool_creation_result["creation_time"],
                    "complexity": tool_creation_result["complexity"]
                })
                collaboration_result["agent_contributions"]["tool_creator"] = tool_creation_result
                
                # Safety Agent's contribution - actually analyze the created code
                if "safety_agent" in involved_agents:
                    safety_agent = self.agents["safety_agent"]
                    
                    safety_report = safety_agent.analyze_code_security(
                        tool_creation_result["code"],
                        context={"task_id": task_id, "divine_message": divine_message}
                    )
                    
                    safety_analysis = {
                        "security_level": safety_report.security_level.value,
                        "risk_score": safety_report.overall_risk_score,
                        "threats_detected": [
                            {
                                "type": threat.threat_type.value,
                                "severity": threat.severity,
                                "description": threat.description
                            }
                            for threat in safety_report.threats_detected
                        ],
                        "approval_status": safety_report.approval_status,
                        "recommendations": safety_report.recommendations,
                        "reasoning": safety_report.reasoning
                    }
                    
                    collaboration_result["safety_analysis"] = safety_analysis
                    collaboration_result["agent_contributions"]["safety_agent"] = safety_analysis
                    
                    # Only proceed with grading if safety approved
                    if safety_report.approval_status and "grading_agent" in involved_agents:
                        grading_agent = self.agents["grading_agent"]
                        
                        # Simulate execution results for grading
                        execution_results = {
                            "success": True,
                            "execution_time": 0.5,  # Would be actual execution time
                            "memory_usage": 5.0     # Would be actual memory usage
                        }
                        
                        performance_report = grading_agent.evaluate_tool_performance(
                            tool_creation_result["code"],
                            tool_creation_result["tool_name"],
                            execution_results=execution_results
                        )
                        
                        performance_eval = {
                            "composite_score": performance_report.composite_score,
                            "grade_letter": performance_report.grade_letter,
                            "strengths": performance_report.strengths,
                            "weaknesses": performance_report.weaknesses,
                            "recommendations": performance_report.recommendations,
                            "metrics": [
                                {
                                    "category": metric.category.value,
                                    "score": metric.score,
                                    "description": metric.description
                                }
                                for metric in performance_report.metrics
                            ]
                        }
                        
                        collaboration_result["performance_evaluation"] = performance_eval
                        collaboration_result["agent_contributions"]["grading_agent"] = performance_eval
                    
                    else:
                        # Safety rejected - no grading performed
                        collaboration_result["performance_evaluation"] = {
                            "message": "Performance evaluation skipped due to safety rejection",
                            "safety_status": "rejected"
                        }
            
            else:
                # Tool creation failed
                collaboration_result["agent_contributions"]["tool_creator"] = {
                    "error": tool_creation_result.get("error", "Tool creation failed"),
                    "message": tool_creation_result.get("message", "Unknown error")
                }
        
        return collaboration_result
    
    async def _complete_task(self, task_record: Dict, result: Dict):
        """Complete a task and update all relevant systems"""
        task_id = task_record["id"]
        
        # Move task to completed
        task_record["status"] = "completed"
        task_record["end_time"] = datetime.now()
        task_record["final_result"] = result
        
        self.completed_tasks.append(task_record)
        del self.active_tasks[task_id]
        
        # Update system metrics
        self.system_metrics["total_tasks"] += 1
        
        # Determine if task was successful
        success = True  # Would be determined by actual evaluation
        if success:
            self.system_metrics["successful_tasks"] += 1
        
        self.system_metrics["tools_created"] += len(result.get("tools_created", []))
        
        # Calculate agent satisfaction
        agent_satisfactions = []
        for agent_name in task_record["assigned_agents"]:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                psychological_state = agent.get_psychological_status()
                agent_satisfactions.append(psychological_state["motivation_summary"]["overall_motivation"])
        
        if agent_satisfactions:
            self.system_metrics["agent_satisfaction"] = sum(agent_satisfactions) / len(agent_satisfactions)
        
        # Create divine feedback
        divine_feedback = {
            "satisfaction_rating": 8 if success else 4,
            "divine_comments": "Your collaborative effort pleases the gods" if success else "More effort required",
            "blessings_granted": ["increased_capabilities"] if success else [],
            "areas_for_improvement": [] if success else ["coordination", "innovation"]
        }
        
        # Send feedback to involved agents
        for agent_name in task_record["assigned_agents"]:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                agent.complete_task(
                    task_record["divine_message"],
                    result,
                    success,
                    divine_feedback
                )
        
        logger.info(f"Task {task_id} completed successfully: {success}")
    
    async def update_universe_systems(self):
        """Update all universe systems - called periodically"""
        
        # Update evolutionary pressure
        pressure_update = update_evolutionary_pressure()
        
        # Check natural laws
        physics_status = check_natural_laws()
        self.system_metrics["universe_stability"] = 1.0 if physics_status["universe_stable"] else 0.5
        
        # Adapt agent drives based on performance
        for agent in self.agents.values():
            psychological_state = agent.get_psychological_status()
            
            # Adapt evolutionary pressure based on agent performance
            performance_data = {
                "success_rate": agent.performance_metrics["success_rate"],
                "motivation_level": psychological_state["motivation_summary"]["overall_motivation"]
            }
            evolutionary_pressure.adapt_difficulty(performance_data)
        
        # Generate intrinsic goals for agents based on current state
        for agent in self.agents.values():
            if agent.drive_system.overall_motivation > 0.7:
                new_goals = agent.drive_system.generate_intrinsic_goals()
                agent.drive_system.active_goals.extend(new_goals)
        
        return {
            "pressure_update": pressure_update,
            "physics_status": physics_status,
            "agent_count": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "system_metrics": self.system_metrics
        }
    
    def get_universe_status(self) -> Dict[str, Any]:
        """Get comprehensive universe status"""
        
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_psychological_status()
        
        return {
            "universe_id": self.universe_id,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "system_metrics": self.system_metrics,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_statuses": agent_statuses,
            "physics_status": check_natural_laws(),
            "pressure_status": evolutionary_pressure.get_current_pressure_status()
        }

# Global orchestrator instance
orchestrator = HyperagenticOrchestrator()

# Background task to update universe systems
async def universe_update_loop():
    """Background loop to continuously update universe systems"""
    while True:
        try:
            await orchestrator.update_universe_systems()
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Error in universe update loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Start background tasks
    update_task = asyncio.create_task(universe_update_loop())
    
    yield
    
    # Cleanup
    update_task.cancel()

# Create main FastAPI app
app = FastAPI(
    title="Hyperagentic Processor",
    description="Organic AGI Development Environment",
    version="1.0.0",
    lifespan=lifespan
)

# Mount divine interface
app.mount("/divine", divine_app)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hyperagentic Processor - Organic AGI Development Environment",
        "universe_id": orchestrator.universe_id,
        "status": "active"
    }

@app.get("/universe/status")
async def get_universe_status():
    """Get comprehensive universe status"""
    return orchestrator.get_universe_status()

@app.post("/universe/divine_task")
async def process_divine_task(message: Dict[str, Any]):
    """Process a divine task through the agent collective"""
    return await orchestrator.process_divine_message(message)

@app.get("/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    statuses = {}
    for name, agent in orchestrator.agents.items():
        statuses[name] = agent.get_psychological_status()
    return statuses

@app.get("/agents/{agent_name}/psychology")
async def get_agent_psychology(agent_name: str):
    """Get detailed psychological profile of a specific agent"""
    if agent_name in orchestrator.agents:
        return orchestrator.agents[agent_name].get_psychological_status()
    else:
        return {"error": f"Agent {agent_name} not found"}

if __name__ == "__main__":
    logger.info("Starting Hyperagentic Processor Universe...")
    
    # Run the universe
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )