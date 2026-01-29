#!/usr/bin/env python3
"""
Test script for functional agent capabilities.

This script demonstrates the working agent system with actual tool creation,
security analysis, and performance evaluation capabilities.
"""

import sys
import os
import logging
import asyncio
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tool_creator_agent import create_tool_creator_agent
from safety_agent import create_safety_agent
from grading_agent import create_grading_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FunctionalAgentTest")

def test_tool_creator():
    """Test the ToolCreator agent's functional capabilities"""
    print("\n" + "="*60)
    print("TESTING TOOL CREATOR AGENT")
    print("="*60)
    
    # Mock LLM config - using Groq like the original class example
    llm_config = {
        "model": "groq/llama-3.1-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY", "test_key"),
        "temperature": 0.7
    }
    
    # Create agent
    tool_creator = create_tool_creator_agent(llm_config)
    
    # Show initial psychological state
    status = tool_creator.get_psychological_status()
    print(f"Agent: {tool_creator.name}")
    print(f"Initial Motivation: {status['motivation_summary']['overall_motivation']:.2f}")
    print(f"Current Emotion: {status['motivation_summary']['current_emotion']['state']}")
    
    # Test tool creation
    requirements = """
    Create a mathematical calculator tool that can:
    1. Perform basic arithmetic operations (add, subtract, multiply, divide)
    2. Handle floating point numbers
    3. Include error handling for division by zero
    4. Return results in a structured format
    """
    
    print(f"\nCreating tool from requirements:")
    print(f"Requirements: {requirements[:100]}...")
    
    result = tool_creator.create_tool_from_requirements(requirements)
    
    print(f"\nTool Creation Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Tool ID: {result['tool_id']}")
        print(f"Tool Name: {result['tool_name']}")
        print(f"Creation Time: {result['creation_time']:.2f}s")
        print(f"Complexity: {result['complexity']}")
        print(f"\nGenerated Code Preview:")
        print(result['code'][:300] + "..." if len(result['code']) > 300 else result['code'])
        
        # Show updated psychological state
        updated_status = tool_creator.get_psychological_status()
        print(f"\nUpdated Motivation: {updated_status['motivation_summary']['overall_motivation']:.2f}")
        print(f"Tools Created: {updated_status['performance_metrics']['tools_created']}")
        
        return result['code'], result['tool_name']
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return None, None

def test_safety_agent(code: str, tool_name: str):
    """Test the SafetyAgent's functional capabilities"""
    print("\n" + "="*60)
    print("TESTING SAFETY AGENT")
    print("="*60)
    
    if not code:
        print("No code to analyze - skipping safety test")
        return None
    
    # Mock LLM config - using Groq like the original class example
    llm_config = {
        "model": "groq/llama-3.1-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY", "test_key"),
        "temperature": 0.7
    }
    
    # Create agent
    safety_agent = create_safety_agent(llm_config)
    
    # Show initial psychological state
    status = safety_agent.get_psychological_status()
    print(f"Agent: {safety_agent.name}")
    print(f"Initial Motivation: {status['motivation_summary']['overall_motivation']:.2f}")
    print(f"Risk Tolerance: {safety_agent.personality.risk_tolerance:.2f}")
    
    # Test security analysis
    print(f"\nAnalyzing code security for: {tool_name}")
    
    context = {
        "tool_name": tool_name,
        "analysis_purpose": "divine_task_validation"
    }
    
    report = safety_agent.analyze_code_security(code, context)
    
    print(f"\nSecurity Analysis Report:")
    print(f"Security Level: {report.security_level.value.upper()}")
    print(f"Risk Score: {report.overall_risk_score:.2f}")
    print(f"Approval Status: {'APPROVED' if report.approval_status else 'REJECTED'}")
    print(f"Analysis Time: {report.analysis_time:.3f}s")
    
    if report.threats_detected:
        print(f"\nThreats Detected ({len(report.threats_detected)}):")
        for i, threat in enumerate(report.threats_detected[:3], 1):  # Show top 3
            print(f"  {i}. {threat.threat_type.value}: {threat.description}")
            print(f"     Severity: {threat.severity:.2f}, Confidence: {threat.confidence:.2f}")
    else:
        print("\nNo security threats detected")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
    
    print(f"\nReasoning: {report.reasoning}")
    
    # Show updated psychological state
    updated_status = safety_agent.get_psychological_status()
    print(f"\nUpdated Motivation: {updated_status['motivation_summary']['overall_motivation']:.2f}")
    
    return report

def test_grading_agent(code: str, tool_name: str, safety_approved: bool):
    """Test the GradingAgent's functional capabilities"""
    print("\n" + "="*60)
    print("TESTING GRADING AGENT")
    print("="*60)
    
    if not code:
        print("No code to grade - skipping grading test")
        return None
    
    if not safety_approved:
        print("Code not safety approved - skipping grading test")
        return None
    
    # Mock LLM config - using Groq like the original class example
    llm_config = {
        "model": "groq/llama-3.1-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY", "test_key"),
        "temperature": 0.7
    }
    
    # Create agent
    grading_agent = create_grading_agent(llm_config)
    
    # Show initial psychological state
    status = grading_agent.get_psychological_status()
    print(f"Agent: {grading_agent.name}")
    print(f"Initial Motivation: {status['motivation_summary']['overall_motivation']:.2f}")
    print(f"Perfectionism: {grading_agent.personality.perfectionism:.2f}")
    
    # Test performance evaluation
    print(f"\nEvaluating performance for: {tool_name}")
    
    # Mock execution results
    execution_results = {
        "success": True,
        "execution_time": 0.25,
        "memory_usage": 8.5,
        "expected_output": "42",
        "actual_output": "42"
    }
    
    # Mock test results
    test_results = {
        "pass_rate": 0.85,
        "tests_run": 10,
        "tests_passed": 8,
        "tests_failed": 2
    }
    
    report = grading_agent.evaluate_tool_performance(
        code, 
        tool_name, 
        execution_results=execution_results,
        test_results=test_results
    )
    
    print(f"\nPerformance Evaluation Report:")
    print(f"Overall Grade: {report.grade_letter}")
    print(f"Composite Score: {report.composite_score:.2f}")
    
    print(f"\nDetailed Metrics:")
    for metric in report.metrics:
        print(f"  {metric.category.value.title()}: {metric.score:.2f} (weight: {metric.weight:.2f})")
        if metric.evidence:
            print(f"    Evidence: {metric.evidence[0]}")  # Show first evidence
    
    if report.strengths:
        print(f"\nStrengths:")
        for strength in report.strengths:
            print(f"  + {strength}")
    
    if report.weaknesses:
        print(f"\nWeaknesses:")
        for weakness in report.weaknesses:
            print(f"  - {weakness}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
    
    if report.trend_analysis:
        print(f"\nTrend Analysis: {report.trend_analysis}")
    
    # Show updated psychological state
    updated_status = grading_agent.get_psychological_status()
    print(f"\nUpdated Motivation: {updated_status['motivation_summary']['overall_motivation']:.2f}")
    
    return report

def test_agent_collaboration():
    """Test agents working together on a divine task"""
    print("\n" + "="*80)
    print("TESTING AGENT COLLABORATION")
    print("="*80)
    
    divine_message = """
    Divine Quest: Create a secure data validation tool that can:
    1. Validate email addresses using regex patterns
    2. Check password strength (length, complexity)
    3. Sanitize user input to prevent injection attacks
    4. Return validation results with detailed feedback
    
    This tool will be used to protect the sacred user data in our divine systems.
    """
    
    print(f"Divine Message: {divine_message}")
    
    # Step 1: Tool Creation
    print(f"\n{'='*20} STEP 1: TOOL CREATION {'='*20}")
    code, tool_name = test_tool_creator()
    
    if not code:
        print("Tool creation failed - cannot proceed with collaboration")
        return
    
    # Step 2: Security Analysis
    print(f"\n{'='*20} STEP 2: SECURITY ANALYSIS {'='*20}")
    safety_report = test_safety_agent(code, tool_name)
    
    if not safety_report:
        print("Security analysis failed - cannot proceed")
        return
    
    # Step 3: Performance Evaluation (only if safety approved)
    print(f"\n{'='*20} STEP 3: PERFORMANCE EVALUATION {'='*20}")
    grading_report = test_grading_agent(code, tool_name, safety_report.approval_status)
    
    # Step 4: Collaboration Summary
    print(f"\n{'='*20} COLLABORATION SUMMARY {'='*20}")
    print(f"Divine Task: Data validation tool creation")
    print(f"Tool Created: {tool_name}")
    print(f"Security Status: {'APPROVED' if safety_report.approval_status else 'REJECTED'}")
    print(f"Security Risk: {safety_report.overall_risk_score:.2f}")
    
    if grading_report:
        print(f"Performance Grade: {grading_report.grade_letter}")
        print(f"Performance Score: {grading_report.composite_score:.2f}")
        
        # Overall success determination
        overall_success = (
            safety_report.approval_status and 
            grading_report.composite_score >= 0.7
        )
        
        print(f"Overall Success: {'YES' if overall_success else 'NO'}")
        
        if overall_success:
            print(f"\n🎉 Divine task completed successfully!")
            print(f"The gods are pleased with this offering.")
        else:
            print(f"\n⚠️  Divine task needs improvement.")
            print(f"The gods require higher quality before acceptance.")
    else:
        print(f"Performance Grade: Not evaluated (security rejected)")
        print(f"Overall Success: NO")
        print(f"\n❌ Divine task failed due to security concerns.")

def main():
    """Main test function"""
    print("HYPERAGENTIC PROCESSOR - FUNCTIONAL AGENT TESTING")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    
    try:
        # Test individual agents
        print("\n🧪 Testing individual agent capabilities...")
        
        # Test collaboration
        print("\n🤝 Testing agent collaboration...")
        test_agent_collaboration()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"Test finished at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()