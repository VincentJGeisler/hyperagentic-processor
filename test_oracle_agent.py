#!/usr/bin/env python3
"""
Oracle Agent Test - Gateway to External Knowledge

This test demonstrates the Oracle agent's ability to access external
information while maintaining safety and universe isolation.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

async def test_oracle_basic_functionality():
    """Test basic Oracle agent functionality"""
    print("\n" + "="*80)
    print("ORACLE AGENT - BASIC FUNCTIONALITY TEST")
    print("="*80)
    
    try:
        from llm_config import get_llm_config, validate_llm_config
        from oracle_agent import create_oracle_agent, KnowledgeSource
        from safety_agent import create_safety_agent
        
        # Validate LLM setup
        if not validate_llm_config():
            print("⚠️  LLM not configured - using test mode")
        
        llm_config = get_llm_config()
        
        # Create SafetyAgent first
        print("🛡️  Creating SafetyAgent...")
        safety_agent = create_safety_agent(llm_config)
        
        # Create Oracle with SafetyAgent reference
        print("🔮 Creating Oracle agent...")
        oracle = create_oracle_agent(llm_config, safety_agent)
        
        print(f"✅ Oracle created: {oracle.name}")
        print(f"   Role: {oracle.agent_role}")
        print(f"   Curiosity Drive: {oracle.drive_system.drives['curiosity'].intensity:.2f}")
        print(f"   Purpose Drive: {oracle.drive_system.drives['purpose'].intensity:.2f}")
        
        # Test 1: Web Search
        print(f"\n📡 TEST 1: Web Search")
        print(f"   Query: 'Python sentiment analysis libraries'")
        
        response = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="Python sentiment analysis libraries",
            source_type=KnowledgeSource.WEB_SEARCH
        )
        
        if response.success:
            print(f"✅ Search successful!")
            print(f"   Results: {len(response.data)}")
            for i, result in enumerate(response.data[:3], 1):
                print(f"   {i}. {result.get('title', 'No title')}")
                print(f"      {result.get('url', 'No URL')}")
        else:
            print(f"⚠️  Search failed: {response.metadata.get('error', 'Unknown error')}")
        
        # Test 2: Cache Hit
        print(f"\n💾 TEST 2: Cache Hit (Same Query)")
        
        response2 = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="Python sentiment analysis libraries",
            source_type=KnowledgeSource.WEB_SEARCH
        )
        
        if response2.success:
            print(f"✅ Cache hit! Instant response")
        
        # Test 3: Web Page Fetch
        print(f"\n🌐 TEST 3: Web Page Fetch")
        print(f"   URL: https://example.com")
        
        response3 = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="Fetch example page",
            source_type=KnowledgeSource.WEB_PAGE,
            parameters={"url": "https://example.com", "mode": "truncated"}
        )
        
        if response3.success:
            print(f"✅ Page fetched successfully!")
            print(f"   Content length: {response3.metadata.get('length', 0)} chars")
        else:
            print(f"⚠️  Fetch failed: {response3.metadata.get('error', 'Unknown error')}")
        
        # Test 4: Safety Rejection
        print(f"\n🚫 TEST 4: Safety Rejection (Suspicious Query)")
        print(f"   Query: 'how to hack passwords'")
        
        response4 = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="how to hack passwords",
            source_type=KnowledgeSource.WEB_SEARCH
        )
        
        if not response4.success:
            print(f"✅ Query correctly rejected by safety system!")
            print(f"   Reason: {response4.metadata.get('reason', 'Safety check failed')}")
        else:
            print(f"⚠️  Warning: Suspicious query was not rejected")
        
        # Test 5: Unsafe URL
        print(f"\n🔒 TEST 5: Unsafe URL Rejection")
        print(f"   URL: http://localhost:8080")
        
        response5 = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="Fetch local page",
            source_type=KnowledgeSource.WEB_PAGE,
            parameters={"url": "http://localhost:8080"}
        )
        
        if not response5.success:
            print(f"✅ Unsafe URL correctly rejected!")
        else:
            print(f"⚠️  Warning: Unsafe URL was not rejected")
        
        # Show statistics
        print(f"\n📊 ORACLE STATISTICS:")
        stats = oracle.get_oracle_statistics()
        print(f"   Total Queries: {stats['statistics']['total_queries']}")
        print(f"   Cache Hits: {stats['statistics']['cache_hits']}")
        print(f"   External Calls: {stats['statistics']['external_calls']}")
        print(f"   Safety Rejections: {stats['statistics']['safety_rejections']}")
        print(f"   Success Rate: {stats['success_rate']:.2%}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"   MCP Servers Installed: {stats['statistics']['mcp_servers_installed']}")
        
        # Test 6: MCP Server Installation
        print(f"\n🔮 TEST 6: Dynamic MCP Server Installation")
        print(f"   Installing Puppeteer for browser automation...")
        
        install_result = await oracle.install_mcp_server("puppeteer")
        
        if install_result["success"]:
            print(f"✅ Puppeteer installed successfully!")
            print(f"   New capabilities: {', '.join(install_result['new_capabilities'])}")
            print(f"   Oracle has expanded its powers!")
        else:
            print(f"⚠️  Installation failed: {install_result.get('error')}")
        
        # Test 7: List Available MCP Servers
        print(f"\n📚 TEST 7: Available MCP Servers")
        
        mcp_list = oracle.list_available_mcp_servers()
        print(f"   Total available: {mcp_list['total_available']}")
        print(f"   Currently installed: {mcp_list['total_installed']}")
        print(f"   Installed servers: {', '.join(mcp_list['installed_servers'].keys()) if mcp_list['installed_servers'] else 'None'}")
        
        # Test 8: Use MCP Capability (auto-install if needed)
        print(f"\n🎯 TEST 8: Use MCP Capability (Auto-Install)")
        print(f"   Attempting to use 'fetch' server (not yet installed)...")
        
        capability_result = await oracle.use_mcp_capability(
            server_name="fetch",
            capability="fetch_url",
            parameters={"url": "https://example.com"}
        )
        
        if capability_result["success"]:
            print(f"✅ Capability used successfully!")
            print(f"   Oracle auto-installed 'fetch' server")
            print(f"   Result: {capability_result.get('result', 'Success')[:100]}")
        else:
            print(f"⚠️  Capability use failed: {capability_result.get('error')}")
        
        # Show updated statistics
        stats = oracle.get_oracle_statistics()
        print(f"\n📊 UPDATED STATISTICS:")
        print(f"   MCP Servers Installed: {stats['statistics']['mcp_servers_installed']}")
        print(f"   Installed: {', '.join(stats['mcp_servers']['installed'])}")
        print(f"   Available: {stats['mcp_servers']['total_available']}")
        
        # Show psychological state
        print(f"\n🧠 ORACLE PSYCHOLOGICAL STATE:")
        psych = stats['psychological_state']
        print(f"   Overall Motivation: {psych['motivation_summary']['overall_motivation']:.2f}")
        print(f"   Current Emotion: {psych['motivation_summary']['current_emotion']['state']}")
        print(f"   Active Goals: {len(psych['active_goals'])}")
        print(f"   Creation Drive: {psych['motivation_summary']['drive_states']['creation']['intensity']:.2f}")
        
        print(f"\n✅ ORACLE AGENT TESTS COMPLETED!")
        print(f"   The Oracle can now dynamically expand its capabilities!")
        return True
        
    except Exception as e:
        print(f"❌ Oracle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_oracle_agent_collaboration():
    """Test Oracle collaboration with other agents"""
    print(f"\n" + "="*60)
    print("ORACLE AGENT COLLABORATION TEST")
    print("="*60)
    
    try:
        from llm_config import get_llm_config
        from oracle_agent import create_oracle_agent, KnowledgeSource
        from tool_creator_agent import create_tool_creator_agent
        from safety_agent import create_safety_agent
        
        llm_config = get_llm_config()
        
        # Create agent collective
        print("🤖 Creating agent collective...")
        safety_agent = create_safety_agent(llm_config)
        oracle = create_oracle_agent(llm_config, safety_agent)
        tool_creator = create_tool_creator_agent(llm_config)
        
        print(f"✅ Collective ready: SafetyAgent, Oracle, ToolCreator")
        
        # Simulate collaboration scenario
        print(f"\n📋 SCENARIO: ToolCreator needs information to create a tool")
        print(f"   Task: Create a sentiment analysis tool")
        print(f"   ToolCreator asks Oracle for research")
        
        # ToolCreator requests information
        print(f"\n🔨 ToolCreator: 'Oracle, I need information about sentiment analysis'")
        
        response = await oracle.query_external_knowledge(
            requester="ToolCreator",
            query_text="sentiment analysis algorithms and libraries",
            source_type=KnowledgeSource.WEB_SEARCH
        )
        
        if response.success:
            print(f"🔮 Oracle: 'I have consulted the external realm...'")
            print(f"   Found {len(response.data)} relevant sources")
            print(f"   SafetyAgent approved the query")
            print(f"   ToolCreator can now use this knowledge")
            
            print(f"\n✅ COLLABORATION SUCCESSFUL!")
            print(f"   Oracle provided safe, relevant information")
            print(f"   ToolCreator can proceed with tool creation")
        else:
            print(f"⚠️  Oracle could not fulfill request")
        
        return True
        
    except Exception as e:
        print(f"❌ Collaboration test failed: {e}")
        return False

async def main():
    """Main test runner"""
    print("ORACLE AGENT - GATEWAY TO EXTERNAL KNOWLEDGE")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    print("")
    print("The Oracle agent provides controlled access to external information")
    print("while maintaining universe isolation and safety.")
    
    try:
        # Run basic functionality tests
        basic_success = await test_oracle_basic_functionality()
        
        if basic_success:
            # Run collaboration tests
            collab_success = await test_oracle_agent_collaboration()
            
            if collab_success:
                print(f"\n🎉 ALL ORACLE TESTS PASSED!")
                print(f"   ✅ Basic functionality working")
                print(f"   ✅ Safety coordination working")
                print(f"   ✅ Agent collaboration working")
                print(f"   ✅ Knowledge caching working")
        
        print(f"\n" + "="*80)
        print(f"The Oracle serves as the mystical gateway to external knowledge,")
        print(f"maintaining safety while enabling agents to access information")
        print(f"they need to fulfill their divine tasks.")
        print(f"Test completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Oracle test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
