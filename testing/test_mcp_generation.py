"""
Test script for MCP Generation System
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import HyperagenticOrchestrator
from src.mcp_generator import MCPRequirements

async def test_mcp_generation():
    """Test the MCP generation system"""
    print("Testing MCP Generation System...")
    
    # Create orchestrator
    orchestrator = HyperagenticOrchestrator()
    
    # Get the Oracle agent
    oracle = orchestrator.agents.get("oracle")
    
    if not oracle:
        print("ERROR: Oracle agent not found")
        return
    
    print("Oracle agent found")
    
    # Test MCP generation
    requirements = MCPRequirements(
        name="test_data_processor",
        description="A simple data processor for testing",
        capability="data processing",
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "operation": {"type": "string"}
            }
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "array"}
            }
        },
        language="python",
        dependencies=[]
    )
    
    print("Generating MCP server...")
    result = await oracle.generate_mcp_server(requirements)
    
    if result.success:
        print(f"SUCCESS: MCP generated successfully!")
        print(f"  MCP Name: {result.mcp_name}")
        print(f"  Output Directory: {result.output_directory}")
        print(f"  Generation Time: {result.generation_time:.2f} seconds")
        print(f"  Validation Passed: {result.validation_passed}")
        if result.code_files:
            print(f"  Code Files: {result.code_files}")
    else:
        print(f"FAILED: MCP generation failed")
        print(f"  Error: {result.error_message}")
    
    # Test find_or_create_mcp
    print("\nTesting find_or_create_mcp...")
    result2 = await oracle.find_or_create_mcp("data processing")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_mcp_generation())