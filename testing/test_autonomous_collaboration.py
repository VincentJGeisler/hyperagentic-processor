"""
Test Autonomous Agent Collaboration

This test demonstrates how agents autonomously collaborate when they
cannot handle tasks alone. It shows:
1. Primary agent determining if it needs help
2. Identifying helper agents based on missing capabilities
3. Agents working together through message passing
4. Results being synthesized from multiple agents
"""

import asyncio
import logging
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import HyperagenticOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutonomousCollaborationTest")


async def test_autonomous_collaboration():
    """Test autonomous agent collaboration"""
    
    logger.info("=" * 80)
    logger.info("AUTONOMOUS AGENT COLLABORATION TEST")
    logger.info("=" * 80)
    
    # Initialize orchestrator
    logger.info("\n1. Initializing orchestrator with agents...")
    orchestrator = HyperagenticOrchestrator()
    
    logger.info(f"Agents initialized: {list(orchestrator.agents.keys())}")
    logger.info(f"Agent registry capabilities:")
    for agent_name, caps in orchestrator.agent_registry.get_all_capabilities().items():
        logger.info(f"  - {agent_name}: {caps}")
    
    # Test Case 1: Task requiring external knowledge
    logger.info("\n" + "=" * 80)
    logger.info("TEST CASE 1: Task Requiring External Knowledge")
    logger.info("=" * 80)
    logger.info("Task: 'What is the origin of Python programming language?'")
    logger.info("Expected: Oracle provides knowledge, ToolCreator synthesizes answer")
    
    task1 = {
        "message_id": "test_001",
        "message": "What is the origin of Python programming language?",
        "priority": 5
    }
    
    result1 = await orchestrator.process_divine_message(task1)
    logger.info(f"\nResult 1 Status: {result1['status']}")
    logger.info(f"Agents Involved: {result1['agents_involved']}")
    
    if "agent_contributions" in result1.get("result", {}):
        logger.info("Agent Contributions:")
        for agent, contribution in result1["result"]["agent_contributions"].items():
            logger.info(f"  - {agent}: {type(contribution)}")
    
    # Test Case 2: Task requiring tool creation
    logger.info("\n" + "=" * 80)
    logger.info("TEST CASE 2: Task Requiring Tool Creation")
    logger.info("=" * 80)
    logger.info("Task: 'Create a function to calculate factorial'")
    logger.info("Expected: ToolCreator creates code, SafetyAgent validates, GradingAgent evaluates")
    
    task2 = {
        "message_id": "test_002",
        "message": "Create a function to calculate factorial of a number",
        "priority": 5
    }
    
    result2 = await orchestrator.process_divine_message(task2)
    logger.info(f"\nResult 2 Status: {result2['status']}")
    logger.info(f"Agents Involved: {result2['agents_involved']}")
    
    if "tools_created" in result2.get("result", {}):
        tools = result2["result"]["tools_created"]
        logger.info(f"Tools Created: {len(tools)}")
        for tool in tools:
            logger.info(f"  - {tool.get('tool_name', 'unknown')}")
    
    # Test Case 3: Complex task requiring multiple capabilities
    logger.info("\n" + "=" * 80)
    logger.info("TEST CASE 3: Complex Task Requiring Multiple Capabilities")
    logger.info("=" * 80)
    logger.info("Task: 'Research the fibonacci sequence and create an optimized implementation'")
    logger.info("Expected: Oracle researches, ToolCreator implements, SafetyAgent validates, GradingAgent evaluates")
    
    task3 = {
        "message_id": "test_003",
        "message": "Research the fibonacci sequence and create an optimized implementation",
        "priority": 8
    }
    
    result3 = await orchestrator.process_divine_message(task3)
    logger.info(f"\nResult 3 Status: {result3['status']}")
    logger.info(f"Agents Involved: {result3['agents_involved']}")
    
    # Test Case 4: Testing can_handle_task directly
    logger.info("\n" + "=" * 80)
    logger.info("TEST CASE 4: Testing can_handle_task() Directly")
    logger.info("=" * 80)
    
    tool_creator = orchestrator.agents["tool_creator"]
    oracle = orchestrator.agents["oracle"]
    
    # Task that ToolCreator can handle alone
    logger.info("\nTest 4a: ToolCreator with code generation task")
    task4a = "Create a simple hello world function"
    can_handle, missing = tool_creator.can_handle_task(task4a)
    logger.info(f"Task: '{task4a}'")
    logger.info(f"ToolCreator can handle: {can_handle}")
    logger.info(f"Missing capabilities: {missing}")
    
    # Task that requires external knowledge
    logger.info("\nTest 4b: ToolCreator with knowledge task")
    task4b = "What is the capital of France?"
    can_handle, missing = tool_creator.can_handle_task(task4b)
    logger.info(f"Task: '{task4b}'")
    logger.info(f"ToolCreator can handle: {can_handle}")
    logger.info(f"Missing capabilities: {missing}")
    
    if missing:
        helpers = tool_creator.identify_helper_agents(missing, orchestrator.agent_registry)
        logger.info(f"Identified helper agents: {helpers}")
    
    # Test Case 5: Testing identify_helper_agents
    logger.info("\n" + "=" * 80)
    logger.info("TEST CASE 5: Testing identify_helper_agents()")
    logger.info("=" * 80)
    
    missing_caps = ["external_knowledge", "web_search"]
    helpers = tool_creator.identify_helper_agents(missing_caps, orchestrator.agent_registry)
    logger.info(f"Missing capabilities: {missing_caps}")
    logger.info(f"Identified helpers: {helpers}")
    
    for helper_name in helpers:
        helper = orchestrator.agents.get(helper_name)
        if helper:
            logger.info(f"  - {helper_name}: {helper.capabilities}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks completed: {orchestrator.system_metrics['total_tasks']}")
    logger.info(f"Successful tasks: {orchestrator.system_metrics['successful_tasks']}")
    logger.info(f"Tools created: {orchestrator.system_metrics['tools_created']}")
    logger.info(f"Agent satisfaction: {orchestrator.system_metrics['agent_satisfaction']:.2f}")
    
    logger.info("\n✅ AUTONOMOUS COLLABORATION TEST COMPLETE")
    logger.info("=" * 80)


async def test_collaboration_workflow():
    """
    Test the specific collaboration workflow:
    1. Primary agent receives task
    2. Primary agent asks: can I handle this alone?
    3. If no, identify helper agents
    4. Request help via messages
    5. Synthesize results
    """
    logger.info("\n" + "=" * 80)
    logger.info("COLLABORATION WORKFLOW TEST")
    logger.info("=" * 80)
    
    orchestrator = HyperagenticOrchestrator()
    
    # Select a primary agent
    task = "Explain quantum computing and create a simulation tool"
    primary_agent_name = orchestrator.select_primary_agent(task)
    logger.info(f"\n1. Primary agent selected: {primary_agent_name}")
    
    primary_agent = orchestrator.agents[primary_agent_name]
    
    # Check if agent can handle task alone
    logger.info(f"\n2. Asking {primary_agent_name} if it can handle task alone...")
    can_handle, missing_capabilities = primary_agent.can_handle_task(task)
    logger.info(f"   Can handle alone: {can_handle}")
    logger.info(f"   Missing capabilities: {missing_capabilities}")
    
    if not can_handle:
        # Identify helper agents
        logger.info(f"\n3. Identifying helper agents for missing capabilities...")
        helper_agents = primary_agent.identify_helper_agents(
            missing_capabilities, 
            orchestrator.agent_registry
        )
        logger.info(f"   Helper agents identified: {helper_agents}")
        
        # Test message flow
        logger.info(f"\n4. Testing message flow...")
        from agent_communication import AgentMessage
        
        # Primary agent sends help request
        help_request = primary_agent.request_help(
            task, 
            missing_capabilities, 
            orchestrator.agent_registry
        )
        logger.info(f"   Help request created:")
        logger.info(f"     From: {help_request.from_agent}")
        logger.info(f"     To: {help_request.to_agent}")
        logger.info(f"     Type: {help_request.type}")
        logger.info(f"     Context: {help_request.context}")
        
        # Helper agents respond
        logger.info(f"\n5. Helper agents responding...")
        for helper_name in helper_agents:
            helper = orchestrator.agents.get(helper_name)
            if helper:
                help_response = helper.handle_message(help_request)
                if help_response:
                    logger.info(f"   Response from {helper_name}:")
                    logger.info(f"     Type: {help_response.type}")
                    logger.info(f"     Content: {help_response.content[:100]}...")
        
        logger.info(f"\n6. Coordination would happen here via coordinate_agent_collaboration()")
        logger.info(f"   This method would:")
        logger.info(f"     - Invoke each helper agent")
        logger.info(f"     - Collect their contributions")
        logger.info(f"     - Primary agent synthesizes results")
        logger.info(f"     - Return final result")
    
    logger.info("\n✅ COLLABORATION WORKFLOW TEST COMPLETE")
    logger.info("=" * 80)


async def main():
    """Run all tests"""
    try:
        # Test 1: Basic autonomous collaboration
        await test_autonomous_collaboration()
        
        # Test 2: Detailed workflow
        await test_collaboration_workflow()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
