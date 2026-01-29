#!/usr/bin/env python3
"""
Comprehensive MCP Ecosystem Testing Script
Tests all components of the MCP ecosystem to verify functionality.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Test configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 30

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.response_data = None
        
    def success(self, details: Dict = None):
        self.passed = True
        if details:
            self.details = details
            
    def failure(self, error: str, details: Dict = None):
        self.passed = False
        self.error = error
        if details:
            self.details = details

class MCPEcosystemTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
        
    async def test_health_endpoint(self) -> TestResult:
        """Test 1: System Health Check"""
        result = TestResult("System Health Check")
        try:
            async with self.session.get(f"{BASE_URL}/health") as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "response": data
                    })
                else:
                    result.failure(f"Health check returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Connection error: {str(e)}")
        return result
    
    async def test_oracle_stats(self) -> TestResult:
        """Test 2: Oracle Statistics Endpoint"""
        result = TestResult("Oracle Statistics")
        try:
            async with self.session.get(f"{BASE_URL}/oracle/stats") as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "total_queries": data.get("total_queries", 0),
                        "web_search_queries": data.get("web_search_queries", 0),
                        "installed_mcps": data.get("installed_mcps", 0)
                    })
                else:
                    result.failure(f"Stats returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_mcp_discovery(self) -> TestResult:
        """Test 3: MCP Discovery"""
        result = TestResult("MCP Discovery")
        try:
            params = {"query": "filesystem", "max_results": 3}
            async with self.session.get(f"{BASE_URL}/oracle/mcp/discover", params=params) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    packages = data.get("packages", [])
                    result.success({
                        "status_code": status,
                        "packages_found": len(packages),
                        "sample_packages": packages[:2] if packages else []
                    })
                else:
                    result.failure(f"Discovery returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_mcp_list(self) -> TestResult:
        """Test 4: List Installed MCPs"""
        result = TestResult("List Installed MCPs")
        try:
            async with self.session.get(f"{BASE_URL}/oracle/mcp/list") as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    mcps = data.get("mcps", {})
                    result.success({
                        "status_code": status,
                        "installed_count": len(mcps),
                        "mcp_names": list(mcps.keys())
                    })
                else:
                    result.failure(f"List returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_oracle_query_auto(self) -> TestResult:
        """Test 5: Oracle Query with AUTO routing (General Information)"""
        result = TestResult("Oracle Query - AUTO routing (General Info)")
        try:
            payload = {
                "query": "what is machine learning",
                "source_type": "auto"
            }
            async with self.session.post(
                f"{BASE_URL}/oracle/query",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "routing_decision": data.get("routing", {}).get("decision"),
                        "capability_detected": data.get("routing", {}).get("capability"),
                        "has_results": bool(data.get("results"))
                    })
                else:
                    result.failure(f"Query returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_oracle_query_web_search(self) -> TestResult:
        """Test 6: Oracle Query with explicit WEB_SEARCH"""
        result = TestResult("Oracle Query - Explicit WEB_SEARCH")
        try:
            payload = {
                "query": "artificial intelligence trends 2026",
                "source_type": "web_search"
            }
            async with self.session.post(
                f"{BASE_URL}/oracle/query",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "has_results": bool(data.get("results")),
                        "result_count": len(data.get("results", [])),
                        "providers_used": data.get("metadata", {}).get("providers_used", [])
                    })
                else:
                    result.failure(f"Query returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_oracle_via_divine(self) -> TestResult:
        """Test 7: Oracle via Divine Interface (Original Issue)"""
        result = TestResult("Oracle via Divine Interface")
        try:
            payload = {
                "message": "ask the oracle to search the web for the origin of cats",
                "priority": 8
            }
            async with self.session.post(
                f"{BASE_URL}/divine/message",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    oracle_knowledge = data.get("oracle_knowledge")
                    result.success({
                        "status_code": status,
                        "oracle_invoked": bool(oracle_knowledge),
                        "has_oracle_knowledge": bool(oracle_knowledge),
                        "response_preview": str(data.get("response", ""))[:100]
                    })
                else:
                    result.failure(f"Divine message returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_oracle_capability_file_ops(self) -> TestResult:
        """Test 8: Oracle AUTO routing - File Operations capability"""
        result = TestResult("Oracle AUTO routing - File Operations")
        try:
            payload = {
                "query": "read and parse a CSV file with sales data",
                "source_type": "auto"
            }
            async with self.session.post(
                f"{BASE_URL}/oracle/query",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "capability_detected": data.get("routing", {}).get("capability"),
                        "routing_decision": data.get("routing", {}).get("decision"),
                        "has_results": bool(data.get("results"))
                    })
                else:
                    result.failure(f"Query returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_oracle_capability_api(self) -> TestResult:
        """Test 9: Oracle AUTO routing - API Integration capability"""
        result = TestResult("Oracle AUTO routing - API Integration")
        try:
            payload = {
                "query": "call a REST API to fetch weather data",
                "source_type": "auto"
            }
            async with self.session.post(
                f"{BASE_URL}/oracle/query",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                if status == 200:
                    result.success({
                        "status_code": status,
                        "capability_detected": data.get("routing", {}).get("capability"),
                        "routing_decision": data.get("routing", {}).get("decision"),
                        "has_results": bool(data.get("results"))
                    })
                else:
                    result.failure(f"Query returned {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def test_error_handling_malformed(self) -> TestResult:
        """Test 10: Error Handling - Malformed Request"""
        result = TestResult("Error Handling - Malformed Request")
        try:
            payload = {
                "invalid_field": "test"
            }
            async with self.session.post(
                f"{BASE_URL}/oracle/query",
                json=payload
            ) as response:
                status = response.status
                data = await response.json()
                result.response_data = data
                
                # We expect an error response (4xx), not a crash
                if 400 <= status < 500:
                    result.success({
                        "status_code": status,
                        "error_handled": True,
                        "error_message": data.get("detail") or data.get("error")
                    })
                elif status == 200:
                    result.failure("Malformed request was accepted (should reject)", {
                        "status_code": status
                    })
                else:
                    result.failure(f"Unexpected status {status}", {
                        "status_code": status,
                        "response": data
                    })
        except Exception as e:
            result.failure(f"Request error: {str(e)}")
        return result
    
    async def run_all_tests(self):
        """Run all tests and generate report"""
        self.print_section("MCP ECOSYSTEM COMPREHENSIVE TEST SUITE")
        print(f"Test started at: {datetime.now().isoformat()}")
        print(f"Base URL: {BASE_URL}")
        print(f"Timeout: {TIMEOUT}s\n")
        
        # Test suite
        tests = [
            self.test_health_endpoint,
            self.test_oracle_stats,
            self.test_mcp_list,
            self.test_mcp_discovery,
            self.test_oracle_query_auto,
            self.test_oracle_query_web_search,
            self.test_oracle_capability_file_ops,
            self.test_oracle_capability_api,
            self.test_oracle_via_divine,
            self.test_error_handling_malformed,
        ]
        
        # Run tests
        for i, test_func in enumerate(tests, 1):
            self.print_section(f"TEST {i}/{len(tests)}: {test_func.__doc__}")
            result = await test_func()
            self.results.append(result)
            
            # Print immediate result
            status_icon = "✓" if result.passed else "✗"
            status_text = "PASSED" if result.passed else "FAILED"
            print(f"{status_icon} {status_text}: {result.name}")
            
            if result.passed:
                print("\nDetails:")
                for key, value in result.details.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nError: {result.error}")
                if result.details:
                    print("\nDetails:")
                    for key, value in result.details.items():
                        print(f"  {key}: {value}")
            
            if result.response_data:
                print(f"\nFull Response:")
                print(json.dumps(result.response_data, indent=2)[:500])
                if len(json.dumps(result.response_data)) > 500:
                    print("  ... (truncated)")
            
            await asyncio.sleep(0.5)  # Brief pause between tests
        
        # Generate summary
        self.print_section("TEST SUMMARY")
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print()
        
        if failed > 0:
            print("Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  ✗ {result.name}: {result.error}")
        
        print(f"\nTest completed at: {datetime.now().isoformat()}")
        
        # Save detailed report
        await self.save_report()
        
        return passed == total
    
    async def save_report(self):
        """Save detailed test report to file"""
        report_file = "test_results.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed)
            },
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "error": r.error,
                    "details": r.details,
                    "response_data": r.response_data
                }
                for r in self.results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {report_file}")

async def main():
    """Main test execution"""
    async with MCPEcosystemTester() as tester:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
