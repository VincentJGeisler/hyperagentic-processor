#!/bin/bash

# MCP Ecosystem Comprehensive Test Script
# Tests all components of the MCP ecosystem

BASE_URL="http://localhost:8001"
OUTPUT_DIR="test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/test_report_${TIMESTAMP}.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize report
echo "==========================================================" > "$REPORT_FILE"
echo "MCP ECOSYSTEM COMPREHENSIVE TEST SUITE" >> "$REPORT_FILE"
echo "Test started at: $(date)" >> "$REPORT_FILE"
echo "Base URL: $BASE_URL" >> "$REPORT_FILE"
echo "==========================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

print_section() {
    local title="$1"
    echo ""
    echo "=========================================="
    echo "  $title"
    echo "=========================================="
    echo "" | tee -a "$REPORT_FILE"
}

run_test() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="${5:-200}"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    print_section "TEST $TESTS_RUN: $test_name"
    echo "Method: $method" | tee -a "$REPORT_FILE"
    echo "Endpoint: $endpoint" | tee -a "$REPORT_FILE"
    
    # Save response to file
    local response_file="${OUTPUT_DIR}/response_${TESTS_RUN}.json"
    local status_code
    
    if [ "$method" = "GET" ]; then
        status_code=$(curl -s -o "$response_file" -w "%{http_code}" "$endpoint")
    else
        status_code=$(curl -s -X "$method" -H "Content-Type: application/json" \
            -d "$data" -o "$response_file" -w "%{http_code}" "$endpoint")
    fi
    
    echo "Status Code: $status_code" | tee -a "$REPORT_FILE"
    
    # Check response
    if [ -f "$response_file" ]; then
        echo "" | tee -a "$REPORT_FILE"
        echo "Response:" | tee -a "$REPORT_FILE"
        cat "$response_file" | python3 -m json.tool 2>/dev/null | head -50 | tee -a "$REPORT_FILE"
        echo "" | tee -a "$REPORT_FILE"
    fi
    
    # Evaluate test
    if [ "$status_code" -ge 200 ] && [ "$status_code" -lt 300 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name" | tee -a "$REPORT_FILE"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name (Status: $status_code)" | tee -a "$REPORT_FILE"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    echo "" | tee -a "$REPORT_FILE"
    sleep 1  # Brief pause between tests
}

# Start testing
print_section "SYSTEM INITIALIZATION"
echo "Checking if server is accessible..." | tee -a "$REPORT_FILE"

# Test 1: Health Check
run_test "System Health Check" \
    "GET" \
    "${BASE_URL}/health"

# Test 2: Oracle Statistics
run_test "Oracle Statistics" \
    "GET" \
    "${BASE_URL}/oracle/stats"

# Test 3: List Installed MCPs
run_test "List Installed MCPs" \
    "GET" \
    "${BASE_URL}/oracle/mcp/list"

# Test 4: MCP Discovery - Filesystem
run_test "MCP Discovery - Filesystem" \
    "GET" \
    "${BASE_URL}/oracle/mcp/discover?query=filesystem&max_results=5"

# Test 5: MCP Discovery - Database
run_test "MCP Discovery - Database" \
    "GET" \
    "${BASE_URL}/oracle/mcp/discover?query=database&max_results=3"

# Test 6: Oracle Query - AUTO routing (General Information)
run_test "Oracle Query - AUTO routing (General Info)" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "what is machine learning", "source_type": "auto"}'

# Test 7: Oracle Query - Explicit WEB_SEARCH
run_test "Oracle Query - Explicit WEB_SEARCH" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "artificial intelligence trends 2026", "source_type": "web_search"}'

# Test 8: Oracle AUTO routing - File Operations
run_test "Oracle AUTO routing - File Operations" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "read and parse a CSV file with sales data", "source_type": "auto"}'

# Test 9: Oracle AUTO routing - API Integration
run_test "Oracle AUTO routing - API Integration" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "call a REST API to fetch weather data", "source_type": "auto"}'

# Test 10: Oracle via Divine Interface (Original Issue)
run_test "Oracle via Divine Interface" \
    "POST" \
    "${BASE_URL}/divine/message" \
    '{"message": "ask the oracle to search the web for the origin of cats", "priority": 8}'

# Test 11: Divine Interface - Web Search Request
run_test "Divine Interface - Web Search Request" \
    "POST" \
    "${BASE_URL}/divine/message" \
    '{"message": "search the web for Python best practices", "priority": 7}'

# Test 12: Oracle Query - Data Processing
run_test "Oracle Query - Data Processing" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "analyze time series data and generate forecast", "source_type": "auto"}'

# Test 13: Error Handling - Malformed Request
run_test "Error Handling - Malformed Request" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"invalid_field": "test"}' \
    "400"

# Test 14: Error Handling - Empty Query
run_test "Error Handling - Empty Query" \
    "POST" \
    "${BASE_URL}/oracle/query" \
    '{"query": "", "source_type": "auto"}' \
    "400"

# Generate Summary
print_section "TEST SUMMARY"
echo "Total Tests Run: $TESTS_RUN" | tee -a "$REPORT_FILE"
echo "Tests Passed: $TESTS_PASSED" | tee -a "$REPORT_FILE"
echo "Tests Failed: $TESTS_FAILED" | tee -a "$REPORT_FILE"

if [ $TESTS_RUN -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($TESTS_PASSED/$TESTS_RUN)*100}")
    echo "Success Rate: ${SUCCESS_RATE}%" | tee -a "$REPORT_FILE"
fi

echo "" | tee -a "$REPORT_FILE"
echo "Test completed at: $(date)" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"
echo "Detailed report saved to: $REPORT_FILE" | tee -a "$REPORT_FILE"
echo "Response files saved to: $OUTPUT_DIR/" | tee -a "$REPORT_FILE"

# Final status
echo ""
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ SOME TESTS FAILED${NC}"
    exit 1
fi
