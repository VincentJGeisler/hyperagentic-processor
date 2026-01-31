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
from agent_registry import AgentRegistry

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
        
        # Initialize Agent Registry
        self.agent_registry = AgentRegistry()
        
        # LLM Configuration - Using Groq like the original class example
        self.llm_config = get_llm_config()
        
        # Validate configuration
        if not validate_llm_config():
            logger.warning("LLM configuration validation failed - using test configuration")
            from llm_config import TEST_LLM_CONFIG
            self.llm_config = TEST_LLM_CONFIG
        
        # Create motivated agents
        self.agents = self._initialize_agents()
        
        # Register agents with their capabilities
        self._register_agents_with_registry()
        
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
        from oracle_agent import create_oracle_agent, OracleAgent, KnowledgeSource
        agents["oracle"] = create_oracle_agent(self.llm_config, agents["safety_agent"])
        
        # Set orchestrator reference in Oracle agent
        agents["oracle"].set_orchestrator(self)
        
        # FIX: Initialize Oracle's async components (web search manager with api_key_available fix)
        # This must be done to populate api_key_available dynamically
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(agents["oracle"].initialize())
        loop.close()
        logger.info("✅ FIX: Oracle agent async initialization completed")
        
        # TODO: Add more agent types as we implement them
        # agents["memory_keeper"] = MotivatedAgentFactory.create_memory_keeper_agent(self.llm_config)
        # agents["reflection_agent"] = MotivatedAgentFactory.create_reflection_agent(self.llm_config)
        # agents["altar_agent"] = MotivatedAgentFactory.create_altar_agent(self.llm_config)
        
        logger.info(f"Initialized {len(agents)} motivated agents with functional capabilities")
        return agents
    
    def _register_agents_with_registry(self):
        """Register all agents with the agent registry system"""
        logger.info("Registering agents with capability registry")
        
        # Register Oracle Agent
        self.agent_registry.register_agent(
            name="oracle",
            capabilities=[
                "external_knowledge",
                "web_search",
                "mcp_discovery",
                "mcp_generation"
            ],
            description="Gateway to external knowledge - searches web, discovers/installs/generates MCP servers",
            agent_instance=self.agents["oracle"],
            metadata={
                "type": "knowledge_agent",
                "primary_function": "external_information_access"
            }
        )
        
        # Register Tool Creator Agent
        self.agent_registry.register_agent(
            name="tool_creator",
            capabilities=[
                "code_generation",
                "tool_creation",
                "text_synthesis"
            ],
            description="Creates functional Python tools and synthesizes text responses",
            agent_instance=self.agents["tool_creator"],
            metadata={
                "type": "creator_agent",
                "primary_function": "tool_generation"
            }
        )
        
        # Register Safety Agent
        self.agent_registry.register_agent(
            name="safety_agent",
            capabilities=[
                "security_analysis",
                "code_validation",
                "threat_detection"
            ],
            description="Analyzes code security and validates safety of generated tools",
            agent_instance=self.agents["safety_agent"],
            metadata={
                "type": "guardian_agent",
                "primary_function": "security_enforcement"
            }
        )
        
        # Register Grading Agent
        self.agent_registry.register_agent(
            name="grading_agent",
            capabilities=[
                "performance_evaluation",
                "quality_assessment",
                "metrics_analysis"
            ],
            description="Evaluates tool performance and provides quality assessments",
            agent_instance=self.agents["grading_agent"],
            metadata={
                "type": "evaluator_agent",
                "primary_function": "performance_measurement"
            }
        )
        
        logger.info(f"Registered {len(self.agents)} agents with {len(self.agent_registry.get_all_capabilities())} capability mappings")
        
        # Log capability distribution
        for agent_name, capabilities in self.agent_registry.get_all_capabilities().items():
            logger.info(f"  - {agent_name}: {capabilities}")
    
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
    
    def select_primary_agent(self, task: str) -> str:
        """
        Select the primary agent using lightweight LLM call (not full AutoGen agent).
        
        FIX ISSUE #2: Uses direct OpenAI API instead of creating full AutoGen agents
        with 300+ line psychological system messages. Reduces tokens by 90%+ while
        maintaining intelligent routing.
        
        Args:
            task: The task description
            
        Returns:
            Name of the primary agent best suited for the task
        """
        logger.info(f"Selecting primary agent for task: {task[:100]}...")
        
        # Get agent info for routing
        agent_info = self.agent_registry.list_all_agents()
        agent_descriptions = "\n".join([
            f"- {agent['name']}: {agent['description']} (Capabilities: {', '.join(agent['capabilities'])})"
            for agent in agent_info
        ])
        
        # Use lightweight OpenAI call (NOT full AutoGen agent with psychological overhead)
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found, falling back to keyword matching")
                return self.agent_registry.find_best_agent_for_task(task) or "tool_creator"
            
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            
            # Minimal routing prompt (not 300+ line psychological message!)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"""Task router: Select best agent for this task.

Available agents:
{agent_descriptions}

Task: {task}

Respond with ONLY the agent name (oracle, tool_creator, safety_agent, or grading_agent):"""
                }],
                temperature=0.3,
                max_tokens=50  # ← Very short response
            )
            
            selected = response.choices[0].message.content.strip().lower()
            
            # Validate and extract
            valid_agents = ["oracle", "tool_creator", "safety_agent", "grading_agent"]
            for agent in valid_agents:
                if agent in selected:
                    logger.info(f"✅ TOKEN-FIX: Selected {agent} via lightweight LLM (~200 tokens)")
                    return agent
            
            # Fallback if parsing failed
            logger.warning(f"Could not parse agent from response: {selected}")
            return "tool_creator"
            
        except Exception as e:
            logger.error(f"Lightweight LLM routing failed: {e}, using keyword fallback")
            return self.agent_registry.find_best_agent_for_task(task) or "tool_creator"
    
    def _analyze_task_requirements(self, divine_text: str) -> List[str]:
        """Analyze divine message to determine which agents should be involved using intelligent routing"""
        logger.info(f"Intelligently analyzing task requirements for: {divine_text[:100]}")
        
        # Select primary agent using LLM analysis
        primary_agent = self.select_primary_agent(divine_text)
        involved_agents = [primary_agent]
        
        # Log the selection
        logger.info(f"Primary agent selected: {primary_agent}")
        
        # Let the primary agent decide if it needs help from other agents
        # This is a simplified version - in Step 6, full agent-to-agent collaboration will be implemented
        # For now, we'll add some common secondary agents based on task type
        text_lower = divine_text.lower()
        
        # If oracle is primary, often need tool_creator to synthesize answers
        if primary_agent == "oracle" and "tool_creator" not in involved_agents:
            involved_agents.append("tool_creator")
            logger.info("Added tool_creator to synthesize Oracle's knowledge into an answer")
        
        # If safety is concerned, add safety agent
        if any(word in text_lower for word in ["safe", "secure", "analyze", "check", "validate"]) and "safety_agent" not in involved_agents:
            involved_agents.append("safety_agent")
            logger.info("Added safety_agent for security concerns")
        
        # If evaluation is mentioned, add grading agent
        if any(word in text_lower for word in ["evaluate", "assess", "grade", "measure", "performance"]) and "grading_agent" not in involved_agents:
            involved_agents.append("grading_agent")
            logger.info("Added grading_agent for evaluation tasks")
        
        logger.info(f"Final agent selection: {involved_agents}")
        return involved_agents
    
    def _detect_oracle_query(self, text_lower: str, original_text: str) -> bool:
        """
        Sophisticated detection of Oracle-type queries using pattern matching and intent classification.
        Returns True if the query requires external knowledge.
        """
        import re
        
        # Pattern 1: Question words at the start (what, why, how, when, where, who)
        question_patterns = [
            r'^what\s+(is|are|was|were|does|do|did|can|could|would|will)',
            r'^why\s+(is|are|was|were|does|do|did|can|could|would|will)',
            r'^how\s+(is|are|was|were|does|do|did|can|could|would|will|to|do)',
            r'^when\s+(is|are|was|were|does|do|did|can|could|would|will)',
            r'^where\s+(is|are|was|were|does|do|did|can|could|would|will)',
            r'^who\s+(is|are|was|were|does|do|did|can|could|would|will)',
            r'^which\s+(is|are|was|were|does|do|did|can|could|would|will)',
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"🔍 PATTERN MATCH: Question pattern detected: {pattern}")
                return True
        
        # Pattern 2: Informational intent keywords (includes computation)
        informational_keywords = [
            "origin", "history", "background", "explain", "describe", "tell me",
            "information about", "details about", "facts about", "learn about",
            "research", "study", "investigate", "explore", "discover",
            "find out", "look up", "search for", "query about",
            "calculate", "compute", "analyze", "process", "evaluate"
        ]
        
        for keyword in informational_keywords:
            if keyword in text_lower:
                logger.info(f"🔍 KEYWORD MATCH: Informational keyword detected: {keyword}")
                return True
        
        # Pattern 3: Explicit external knowledge requests
        external_keywords = [
            "oracle", "search", "web", "find", "lookup", "knowledge",
            "external", "query", "internet", "online", "browse"
        ]
        
        for keyword in external_keywords:
            if keyword in text_lower:
                logger.info(f"🔍 EXPLICIT: External knowledge keyword detected: {keyword}")
                return True
        
        # Pattern 4: Factual query patterns (noun phrases asking for facts)
        factual_patterns = [
            r'(cause|causes)\s+of',
            r'(reason|reasons)\s+for',
            r'(definition|meaning)\s+of',
            r'(inventor|creator|founder)\s+of',
            r'(date|time|year|period)\s+(of|when)',
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"🔍 FACTUAL PATTERN: Factual query pattern detected: {pattern}")
                return True
        
        # Pattern 5: Questions ending with '?'
        if original_text.strip().endswith('?'):
            # Check if it's likely a factual question (not a code/tool creation question)
            non_oracle_keywords = ["create", "build", "implement", "generate", "develop", "code", "write", "make"]
            if not any(keyword in text_lower for keyword in non_oracle_keywords):
                logger.info(f"🔍 QUESTION MARK: Factual question detected (ends with ?)")
                return True
        
        logger.info(f"❌ NO ORACLE MATCH: Query does not require external knowledge")
        return False
    
    async def _coordinate_agent_collaboration(self, task_record: Dict) -> Dict[str, Any]:
        """
        Coordinate multiple agents to collaboratively address a divine task.
        
        This is where the magic happens - agents with different drives and
        personalities work together, each contributing their unique perspective.
        """
        task_id = task_record["id"]
        divine_message = task_record["divine_message"]
        involved_agents = task_record["assigned_agents"]
        
        # DIAGNOSTIC: Log which agents are involved
        logger.info(f"🐛 DEBUG: Coordinating collaboration for task {task_id}")
        logger.info(f"🐛 DEBUG: Involved agents: {involved_agents}")
        logger.info(f"🐛 DEBUG: Divine message: {divine_message[:100]}")
        
        # Enable autonomous collaboration by letting the primary agent decide
        primary_agent_name = involved_agents[0] if involved_agents else "tool_creator"
        primary_agent = self.agents.get(primary_agent_name)
        
        if primary_agent:
            # Ask primary agent if it can handle the task alone
            logger.info(f"Asking primary agent {primary_agent_name} if it can handle task alone")
            can_handle, missing_capabilities = primary_agent.can_handle_task(divine_message)
            
            if can_handle:
                logger.info(f"Primary agent {primary_agent_name} can handle task alone")
                # Let primary agent handle the task
                return await self._handle_task_with_agent(primary_agent, divine_message, task_id)
            else:
                logger.info(f"Primary agent {primary_agent_name} needs help with: {missing_capabilities}")
                # Facilitate collaboration between agents
                helper_agents = primary_agent.identify_helper_agents(missing_capabilities, self.agent_registry)
                logger.info(f"Identified helper agents: {helper_agents}")
                
                # Coordinate collaboration using either sequential or GroupChat approach
                return await self.coordinate_agent_collaboration(primary_agent, divine_message, helper_agents, task_id)
        else:
            # Fallback to original coordination method
            return await self._coordinate_agent_collaboration_original(task_record)
    
    async def _coordinate_agent_collaboration_original(self, task_record: Dict) -> Dict[str, Any]:
        """
        Original coordination method for backward compatibility.
        """
        task_id = task_record["id"]
        divine_message = task_record["divine_message"]
        involved_agents = task_record["assigned_agents"]
        
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
        
        # Oracle Agent's contribution - query external knowledge
        if "oracle" in involved_agents:
            logger.info(f"Consulting Oracle for external knowledge: {divine_message[:100]}")
            oracle = self.agents["oracle"]
            try:
                from oracle_agent import KnowledgeSource
                oracle_response = await oracle.query_external_knowledge(
                    requester="DivineInterface",
                    query_text=divine_message,
                    source_type=KnowledgeSource.AUTO,  # Changed from WEB_SEARCH to AUTO
                    parameters={"max_results": 10}
                )
                
                # DIAGNOSTIC: Log what Oracle returned
                logger.info(f"🔍 TRACE-4: main.py received oracle_response")
                logger.info(f"🔍 TRACE-4: oracle_response.success = {oracle_response.success}")
                logger.info(f"🔍 TRACE-4: oracle_response.data type = {type(oracle_response.data)}")
                logger.info(f"🔍 TRACE-4: oracle_response.data length = {len(oracle_response.data) if isinstance(oracle_response.data, list) else 'not a list'}")
                if isinstance(oracle_response.data, list) and oracle_response.data:
                    logger.info(f"🔍 TRACE-4: First item in oracle_response.data: {oracle_response.data[0]}")
                
                collaboration_result["oracle_knowledge"] = oracle_response.data
                
                # DIAGNOSTIC: Verify assignment
                logger.info(f"🔍 TRACE-4: After assignment, collaboration_result['oracle_knowledge'] = {type(collaboration_result['oracle_knowledge'])}")
                logger.info(f"🔍 TRACE-4: collaboration_result['oracle_knowledge'] length = {len(collaboration_result['oracle_knowledge']) if isinstance(collaboration_result['oracle_knowledge'], list) else 'not a list'}")
                
                collaboration_result["agent_contributions"]["oracle"] = {
                    "knowledge": oracle_response.data,
                    "source": oracle_response.source,
                    "success": oracle_response.success,
                    "metadata": oracle_response.metadata
                }
                
                # DIAGNOSTIC: Log Oracle response details
                logger.info(f"🐛 DEBUG: Oracle returned knowledge: {len(str(oracle_response.data))} chars")
                logger.info(f"🐛 DEBUG: Oracle success: {oracle_response.success}")
                logger.info(f"🐛 DEBUG: ToolCreator in involved_agents? {'tool_creator' in involved_agents}")
                
                # FIX #2: Use ToolCreator's text synthesis method (NOT code generation)
                if oracle_response.success and "tool_creator" in involved_agents:
                    logger.info(f"✅ FIX: Synthesizing Oracle knowledge into readable answer")
                    tool_creator = self.agents["tool_creator"]
                    
                    # Format the knowledge for synthesis
                    if isinstance(oracle_response.data, list) and len(oracle_response.data) > 0:
                        # Format search results nicely
                        formatted_knowledge = "\n\n".join([
                            f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                            f"URL: {result.get('url', 'N/A')}\n"
                            f"Content: {result.get('snippet', 'No content')}"
                            for i, result in enumerate(oracle_response.data[:5])  # Use top 5 results
                        ])
                    else:
                        formatted_knowledge = json.dumps(oracle_response.data, indent=2)
                    
                    # Create synthesis prompt for text generation (not code)
                    synthesis_prompt = f"""You are synthesizing information to answer a user's question.

Question: {divine_message}

External Knowledge Sources:
{formatted_knowledge}

Task: Based on the external knowledge above, provide a clear, comprehensive, and well-structured answer to the user's question. Your response should:
1. Directly answer the question
2. Include relevant details from the sources
3. Be written in natural language (NOT code)
4. Cite sources when mentioning specific facts
5. Be concise but thorough

Provide your answer below:"""
                    
                    # Call ToolCreator's text synthesis method (preserves code generation ability)
                    synthesis_result = tool_creator.synthesize_text_response(
                        prompt=synthesis_prompt,
                        context={"oracle_data": oracle_response.data}
                    )
                    
                    if synthesis_result["success"]:
                        collaboration_result["synthesized_answer"] = synthesis_result["text"]
                        collaboration_result["agent_contributions"]["tool_creator_synthesis"] = {
                            "answer": synthesis_result["text"],
                            "based_on_oracle_knowledge": True,
                            "method": synthesis_result["method"],
                            "sources_used": len(oracle_response.data) if isinstance(oracle_response.data, list) else 0
                        }
                        logger.info(f"✅ FIX: Successfully synthesized readable answer from Oracle knowledge")
                    else:
                        logger.error(f"⚠️ Text synthesis failed: {synthesis_result.get('error')}")
                        # Fallback: provide formatted summary
                        collaboration_result["synthesized_answer"] = f"Answer based on external research:\n\n{formatted_knowledge}"
                else:
                    logger.warning(f"⚠️ Cannot synthesize: oracle_success={oracle_response.success}, tool_creator_available={'tool_creator' in involved_agents}")
                
            except Exception as e:
                logger.error(f"Oracle query failed: {e}")
                collaboration_result["oracle_error"] = str(e)
                collaboration_result["agent_contributions"]["oracle"] = {
                    "error": str(e)
                }
        
        # DIAGNOSTIC: Log final collaboration result structure
        logger.info(f"🔍 TRACE-5: Final collaboration_result keys: {list(collaboration_result.keys())}")
        logger.info(f"🔍 TRACE-5: Agent contributions: {list(collaboration_result['agent_contributions'].keys())}")
        
        # DIAGNOSTIC: Check oracle_knowledge specifically
        if "oracle_knowledge" in collaboration_result:
            logger.info(f"🔍 TRACE-5: oracle_knowledge exists in result")
            logger.info(f"🔍 TRACE-5: oracle_knowledge type: {type(collaboration_result['oracle_knowledge'])}")
            logger.info(f"🔍 TRACE-5: oracle_knowledge length: {len(collaboration_result['oracle_knowledge']) if isinstance(collaboration_result['oracle_knowledge'], list) else 'not a list'}")
            if isinstance(collaboration_result['oracle_knowledge'], list) and collaboration_result['oracle_knowledge']:
                logger.info(f"🔍 TRACE-5: First oracle_knowledge item: {collaboration_result['oracle_knowledge'][0]}")
        else:
            logger.error(f"🔍 TRACE-5: oracle_knowledge NOT in collaboration_result!")
        
        return collaboration_result
    
    async def _handle_task_with_agent(self, agent, task: str, task_id: str) -> Dict[str, Any]:
        """
        Handle a task with a single agent.
        
        Args:
            agent: The agent to handle the task
            task: The task description
            task_id: The task identifier
            
        Returns:
            Result dictionary from the agent's work
        """
        logger.info(f"🔧 CONTROL FLOW: _handle_task_with_agent called with agent={agent.name}")
        logger.info(f"🔧 CONTROL FLOW: Task={task[:100]}")
        
        # Base result structure - NO EARLY RETURN HERE
        result = {
            "agent": agent.name,
            "task_handled": task,
            "timestamp": datetime.now().isoformat()
        }
        
        # CRITICAL: Agent-specific handling MUST execute before returning
        # Each branch populates the result dict, then we return at the end
        
        agent_name_normalized = agent.name.lower().replace("_", "")
        
        if agent_name_normalized == "toolcreator":
            logger.info(f"🔧 CONTROL FLOW: Entering ToolCreator branch")
            # Check if this is a tool creation request or a simple question
            task_lower = task.lower()
            is_tool_request = any(keyword in task_lower for keyword in [
                "create", "build", "generate", "implement", "develop", "write code", "make a tool"
            ])
            
            if is_tool_request:
                # Handle as tool creation request
                tool_result = agent.create_tool_from_requirements(task)
                result["tool_creation"] = tool_result
                result["result"] = tool_result.get("message", "Tool creation completed")
            else:
                # Handle as a question that needs an answer
                logger.info(f"ToolCreator answering question directly: {task[:100]}")
                answer_prompt = f"""Answer this question directly and concisely:

Question: {task}

Provide a clear, helpful answer:"""
                
                answer_result = agent.synthesize_text_response(
                    prompt=answer_prompt,
                    context={}
                )
                
                if answer_result.get("success"):
                    result["result"] = answer_result["text"]
                    result["answer"] = answer_result["text"]
                    result["method"] = "direct_answer"
                else:
                    # Fallback to simple answer
                    result["result"] = "I can help you with that task."
                    result["answer"] = "I can help you with that task."
        elif agent_name_normalized == "oracle":
            logger.info(f"🔍 DEBUG: Entering Oracle branch for task: {task[:100]}")
            from oracle_agent import KnowledgeSource
            oracle_result = await agent.query_external_knowledge(
                requester="Orchestrator",
                query_text=task,
                source_type=KnowledgeSource.AUTO,
                parameters={"max_results": 5}
            )
            logger.info(f"🔍 DEBUG: Oracle query completed - success={oracle_result.success}, data_length={len(str(oracle_result.data))}")
            result["oracle_query"] = {
                "success": oracle_result.success,
                "data": oracle_result.data,
                "source": oracle_result.source
            }
            
            # FIX: Synthesize Oracle knowledge into human-readable answer
            if oracle_result.success and oracle_result.data:
                tool_creator = self.agents.get("tool_creator")
                if tool_creator:
                    # Format the knowledge for synthesis
                    if isinstance(oracle_result.data, list) and len(oracle_result.data) > 0:
                        formatted_knowledge = "\n\n".join([
                            f"Source {i+1}: {item.get('title', 'Untitled')}\n"
                            f"URL: {item.get('url', 'N/A')}\n"
                            f"Content: {item.get('snippet', 'No content')}"
                            for i, item in enumerate(oracle_result.data[:5])
                        ])
                    else:
                        formatted_knowledge = json.dumps(oracle_result.data, indent=2)
                    
                    synthesis_prompt = f"""Answer this question using the external knowledge provided.

Question: {task}

External Knowledge Sources:
{formatted_knowledge}

Provide a clear, comprehensive answer based on the sources above:"""
                    
                    synthesis_result = tool_creator.synthesize_text_response(
                        prompt=synthesis_prompt,
                        context={"oracle_data": oracle_result.data}
                    )
                    
                    if synthesis_result.get("success"):
                        result["answer"] = synthesis_result["text"]
                        result["method"] = "oracle_with_synthesis"
                        logger.info(f"✅ Oracle query synthesized into human-readable answer")
                    else:
                        # Fallback to formatted knowledge
                        result["answer"] = f"Based on external research:\n\n{formatted_knowledge}"
                        result["method"] = "oracle_formatted"
                else:
                    # No tool_creator available, provide formatted summary
                    if isinstance(oracle_result.data, list) and len(oracle_result.data) > 0:
                        summary = "\n".join([f"• {item.get('title', 'Result')}: {item.get('snippet', '')[:200]}"
                                           for item in oracle_result.data[:3]])
                        result["answer"] = f"Found information:\n{summary}"
                    else:
                        result["answer"] = str(oracle_result.data)
                    result["method"] = "oracle_raw"
        elif agent_name_normalized == "safetyagent":
            # Safety agent needs code to analyze, so we'll create a simple test
            test_code = "# Simple test code\nprint('Hello, World!')"
            safety_result = agent.analyze_code_security(test_code)
            result["safety_analysis"] = {
                "security_level": safety_result.security_level.value,
                "risk_score": safety_result.overall_risk_score,
                "approval_status": safety_result.approval_status
            }
        elif agent_name_normalized == "gradingagent":
            # Grading agent needs code to evaluate, so we'll create a simple test
            test_code = "# Simple test code\ndef hello():\n    return 'Hello, World!'"
            grading_result = agent.evaluate_tool_performance(test_code, "hello_function")
            result["performance_evaluation"] = {
                "composite_score": grading_result.composite_score,
                "grade_letter": grading_result.grade_letter
            }
        
        return result
    
    async def coordinate_agent_collaboration(
        self,
        primary_agent: Any,
        task: str,
        helper_agents: List[str],
        task_id: str
    ) -> Dict[str, Any]:
        """
        Coordinate collaboration between primary agent and helper agents.
        
        This implements autonomous agent collaboration where agents work together
        to accomplish tasks they cannot handle alone.
        
        Args:
            primary_agent: The primary agent handling the task
            task: The task to be accomplished
            helper_agents: List of helper agent names
            task_id: Task identifier for tracking
            
        Returns:
            Combined result from agent collaboration
        """
        logger.info(f"🤝 AUTONOMOUS COLLABORATION: Primary agent {primary_agent.name} requesting help from {helper_agents}")
        
        # Approach 1: Sequential Collaboration
        # Let each helper agent contribute, then primary agent synthesizes
        helper_results = {}
        
        for helper_name in helper_agents:
            helper_agent = self.agents.get(helper_name)
            if not helper_agent:
                logger.warning(f"Helper agent {helper_name} not found")
                continue
            
            logger.info(f"🤝 Requesting help from {helper_name}")
            
            # Create a help request message
            from agent_communication import AgentMessage
            help_request = primary_agent.request_help(task, [helper_name], self.agent_registry)
            
            # Handle the message with the helper agent
            help_response = helper_agent.handle_message(help_request)
            
            # Get actual work from the helper agent
            helper_result = await self._get_helper_contribution(helper_agent, task, task_id)
            helper_results[helper_name] = helper_result
            
            logger.info(f"✅ Received help from {helper_name}")
        
        # Primary agent synthesizes all results
        logger.info(f"🎯 Primary agent {primary_agent.name} synthesizing results from {len(helper_results)} helpers")
        
        final_result = await self._synthesize_collaboration_results(
            primary_agent,
            task,
            helper_results,
            task_id
        )
        
        logger.info(f"✅ AUTONOMOUS COLLABORATION COMPLETE: Task handled with {len(helper_results)} helper agents")
        
        return final_result
    
    async def _get_helper_contribution(
        self,
        helper_agent: Any,
        task: str,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Get the actual contribution from a helper agent.
        
        Args:
            helper_agent: The helper agent
            task: The task description
            task_id: Task identifier
            
        Returns:
            The helper agent's contribution
        """
        logger.info(f"Getting contribution from {helper_agent.name}")
        
        # Different agents contribute in different ways
        helper_name_normalized = helper_agent.name.lower().replace("_", "")
        if helper_name_normalized == "oracle":
            from oracle_agent import KnowledgeSource
            oracle_response = await helper_agent.query_external_knowledge(
                requester="Collaboration",
                query_text=task,
                source_type=KnowledgeSource.AUTO,
                parameters={"max_results": 5}
            )
            return {
                "agent": "oracle",
                "contribution_type": "external_knowledge",
                "success": oracle_response.success,
                "data": oracle_response.data,
                "source": oracle_response.source
            }
        
        elif helper_name_normalized == "toolcreator":
            tool_result = helper_agent.create_tool_from_requirements(task)
            return {
                "agent": "tool_creator",
                "contribution_type": "tool_creation",
                "success": tool_result["success"],
                "data": tool_result
            }
        
        elif helper_name_normalized == "safetyagent":
            # Safety agent needs code to analyze
            return {
                "agent": "safety_agent",
                "contribution_type": "security_analysis",
                "ready": True,
                "message": "Ready to analyze code when provided"
            }
        
        elif helper_name_normalized == "gradingagent":
            # Grading agent needs code to evaluate
            return {
                "agent": "grading_agent",
                "contribution_type": "performance_evaluation",
                "ready": True,
                "message": "Ready to evaluate performance when provided"
            }
        
        else:
            return {
                "agent": helper_agent.name,
                "contribution_type": "general",
                "message": f"Contribution from {helper_agent.name}"
            }
    
    async def _synthesize_collaboration_results(
        self,
        primary_agent: Any,
        task: str,
        helper_results: Dict[str, Dict],
        task_id: str
    ) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents into a final result.
        
        Args:
            primary_agent: The primary agent
            task: The original task
            helper_results: Results from helper agents
            task_id: Task identifier
            
        Returns:
            Synthesized final result
        """
        logger.info(f"Synthesizing results from {len(helper_results)} agents")
        
        synthesis_result = {
            "collaboration_type": "autonomous",
            "primary_agent": primary_agent.name,
            "helper_agents": list(helper_results.keys()),
            "task": task,
            "task_id": task_id,
            "agent_contributions": helper_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # If oracle provided knowledge and tool_creator is available, synthesize
        if "oracle" in helper_results and helper_results["oracle"].get("success"):
            oracle_data = helper_results["oracle"].get("data", [])
            
            primary_name_normalized = primary_agent.name.lower().replace("_", "")
            if primary_name_normalized == "toolcreator":
                # Tool creator can synthesize the knowledge
                if isinstance(oracle_data, list) and len(oracle_data) > 0:
                    formatted_knowledge = "\n\n".join([
                        f"Source {i+1}: {result.get('title', 'Untitled')}\n"
                        f"URL: {result.get('url', 'N/A')}\n"
                        f"Content: {result.get('snippet', 'No content')}"
                        for i, result in enumerate(oracle_data[:5])
                    ])
                else:
                    formatted_knowledge = json.dumps(oracle_data, indent=2)
                
                synthesis_prompt = f"""You are synthesizing information to answer a user's question.

Question: {task}

External Knowledge Sources:
{formatted_knowledge}

Task: Provide a clear, comprehensive, and well-structured answer based on the external knowledge above.
"""
                
                synthesis_text = primary_agent.synthesize_text_response(
                    prompt=synthesis_prompt,
                    context={"oracle_data": oracle_data}
                )
                
                if synthesis_text.get("success"):
                    synthesis_result["synthesized_answer"] = synthesis_text["text"]
                    synthesis_result["synthesis_method"] = "tool_creator_with_oracle"
        
        # If tool was created, add safety analysis if available
        if "tool_creator" in helper_results and "safety_agent" in helper_results:
            tool_data = helper_results["tool_creator"].get("data", {})
            if tool_data.get("success") and tool_data.get("code"):
                safety_agent = self.agents.get("safety_agent")
                if safety_agent:
                    safety_report = safety_agent.analyze_code_security(
                        tool_data["code"],
                        context={"task_id": task_id, "collaboration": True}
                    )
                    synthesis_result["safety_analysis"] = {
                        "security_level": safety_report.security_level.value,
                        "risk_score": safety_report.overall_risk_score,
                        "approval_status": safety_report.approval_status,
                        "reasoning": safety_report.reasoning
                    }
        
        # Add collaboration metrics
        synthesis_result["collaboration_metrics"] = {
            "total_agents": len(helper_results) + 1,  # +1 for primary
            "successful_contributions": sum(1 for r in helper_results.values() if r.get("success", False)),
            "collaboration_time": datetime.now().isoformat()
        }
        
        logger.info(f"Synthesis complete with {synthesis_result['collaboration_metrics']['successful_contributions']} successful contributions")
        
        return synthesis_result
    
    async def coordinate_agent_collaboration_with_groupchat(
        self,
        primary_agent: Any,
        task: str,
        helper_agents: List[str],
        task_id: str
    ) -> Dict[str, Any]:
        """
        Coordinate collaboration using AutoGen GroupChat.
        
        This is an alternative, more powerful approach that lets agents
        autonomously discuss and collaborate through a group chat.
        
        Args:
            primary_agent: The primary agent handling the task
            task: The task to be accomplished
            helper_agents: List of helper agent names
            task_id: Task identifier
            
        Returns:
            Combined result from GroupChat collaboration
        """
        logger.info(f"🎭 GROUPCHAT COLLABORATION: Starting group chat with {[primary_agent.name] + helper_agents}")
        
        try:
            from autogen import GroupChat, GroupChatManager
            
            # Gather all agent instances
            all_agents = [primary_agent]
            for helper_name in helper_agents:
                helper = self.agents.get(helper_name)
                if helper:
                    all_agents.append(helper)
            
            logger.info(f"GroupChat participants: {[a.name for a in all_agents]}")
            
            # Create GroupChat
            groupchat = GroupChat(
                agents=all_agents,
                messages=[],
                max_round=10,
                speaker_selection_method="auto"  # Agents decide who speaks next
            )
            
            # Create GroupChat Manager
            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config=self.llm_config
            )
            
            # Initiate the collaboration
            logger.info(f"Initiating group chat for task: {task[:100]}...")
            
            # Primary agent starts the discussion
            chat_result = primary_agent.initiate_chat(
                manager,
                message=f"Task: {task}\n\nI need help with this task. Let's collaborate to solve it.",
                silent=False
            )
            
            # Extract results from chat history
            result = {
                "collaboration_type": "groupchat",
                "primary_agent": primary_agent.name,
                "participants": [a.name for a in all_agents],
                "task": task,
                "task_id": task_id,
                "chat_history": chat_result.chat_history if hasattr(chat_result, 'chat_history') else [],
                "total_rounds": len(chat_result.chat_history) if hasattr(chat_result, 'chat_history') else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ GROUPCHAT COLLABORATION COMPLETE: {result['total_rounds']} rounds of discussion")
            
            return result
            
        except Exception as e:
            logger.error(f"GroupChat collaboration failed: {e}")
            # Fallback to sequential collaboration
            logger.info("Falling back to sequential collaboration")
            return await self.coordinate_agent_collaboration(primary_agent, task, helper_agents, task_id)
    
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
    result = await orchestrator.process_divine_message(message)
    
    # FIX ISSUE #1: Extract human-readable answer to top level for user-friendly response
    final_result = result.get("result", {})
    
    # Priority 1: Check for synthesized_answer (from collaboration)
    if isinstance(final_result, dict) and "synthesized_answer" in final_result:
        return {
            "task_id": result.get("task_id"),
            "status": "completed",
            "answer": final_result["synthesized_answer"],  # ← Main answer at top level
            "agents_involved": result.get("agents_involved", []),
            "completion_time": result.get("completion_time"),
            "metadata": {
                "oracle_sources": len(final_result.get("oracle_knowledge", [])),
                "agent_contributions": list(final_result.get("agent_contributions", {}).keys())
            }
        }
    # Priority 2: Check for direct answer field (from single agent)
    elif isinstance(final_result, dict) and "answer" in final_result:
        return {
            "task_id": result.get("task_id"),
            "status": "completed",
            "answer": final_result["answer"],  # ← Direct answer from agent
            "agents_involved": result.get("agents_involved", []),
            "completion_time": result.get("completion_time"),
            "metadata": {
                "method": final_result.get("method", "direct"),
                "agent": final_result.get("agent", "unknown")
            }
        }
    # Priority 3: Check for result field with text answer
    elif isinstance(final_result, dict) and "result" in final_result and isinstance(final_result["result"], str):
        return {
            "task_id": result.get("task_id"),
            "status": "completed",
            "answer": final_result["result"],  # ← Result as string answer
            "agents_involved": result.get("agents_involved", []),
            "completion_time": result.get("completion_time"),
            "metadata": {
                "agent": final_result.get("agent", "unknown")
            }
        }
    else:
        # Fallback: return full result if no human-readable answer found
        return result

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