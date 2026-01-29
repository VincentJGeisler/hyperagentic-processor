# Oracle Agent - Gateway to External Knowledge

## Overview

The Oracle Agent is a special agent that serves as the **controlled gateway** between the isolated agent universe and the external world. From the agents' perspective, the Oracle is a mystical entity with access to knowledge beyond their universe's boundaries.

## The Oracle Metaphor

In the agent universe:
- **Agents are isolated** - They cannot directly access external information
- **The Oracle is mystical** - A being with powers beyond normal agents
- **Knowledge is sacred** - Information from the Oracle comes from "the external realm"
- **Safety is paramount** - The Oracle works with the SafetyAgent to ensure information is safe

This maintains the universe's isolation while providing necessary external knowledge access.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External World                            │
│  - Web Search (Google, etc.)                                │
│  - Web Pages                                                 │
│  - Documents (PDF, DOC, etc.)                               │
│  - APIs                                                      │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ MCP Tools
                            │
┌───────────────────────────┼─────────────────────────────────┐
│         Agent Universe    │                                  │
│                           │                                  │
│  ┌────────────────────────▼──────────────────────────────┐  │
│  │           Oracle Agent                                │  │
│  │  - Web search capability                              │  │
│  │  - Web page fetching                                  │  │
│  │  - Document processing                                │  │
│  │  - Knowledge caching                                  │  │
│  │  - Safety coordination                                │  │
│  └────────────────────────┬──────────────────────────────┘  │
│                           │                                  │
│                           │ Consults                         │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         SafetyAgent                                 │    │
│  │  - Approves/rejects queries                        │    │
│  │  - Checks for suspicious patterns                  │    │
│  │  - Validates URLs                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           │ Provides Knowledge               │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    Other Agents (ToolCreator, etc.)                │    │
│  │  - Request knowledge from Oracle                   │    │
│  │  - Receive formatted information                   │    │
│  │  - Use knowledge for tasks                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Capabilities

### 0. Dynamic MCP Server Installation (Special Ability)

**The Oracle's most powerful ability**: It can install new MCP servers to expand its own capabilities.

```python
# Oracle realizes it needs browser automation
result = await oracle.install_mcp_server("puppeteer")

# Now Oracle can use Puppeteer capabilities
screenshot = await oracle.use_mcp_capability(
    server_name="puppeteer",
    capability="screenshot",
    parameters={"url": "https://example.com", "name": "example"}
)
```

**Available MCP Servers:**
- **puppeteer**: Browser automation, screenshots, navigation
- **fetch**: Advanced web content fetching
- **filesystem**: File operations
- **postgres**: PostgreSQL database access
- **sqlite**: SQLite database access
- **github**: GitHub repository access
- **google-maps**: Google Maps API
- **memory**: Persistent memory storage

**Features:**
- Automatic installation when capability is needed
- Safety approval from SafetyAgent
- Capability verification before use
- Installation tracking and statistics
- Dynamic capability expansion

### 1. Web Search
```python
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="sentiment analysis algorithms",
    source_type=KnowledgeSource.WEB_SEARCH
)
```

**Features:**
- Search major search engines
- Return top results with titles, URLs, snippets
- Cache results to reduce external calls
- Safety approval required

### 2. Web Page Fetching
```python
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Fetch documentation page",
    source_type=KnowledgeSource.WEB_PAGE,
    parameters={"url": "https://example.com/docs", "mode": "full"}
)
```

**Features:**
- Fetch full or truncated page content
- Parse HTML to readable text
- Validate URLs for safety
- Block localhost and private IPs

### 3. Document Processing
```python
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Download API specification",
    source_type=KnowledgeSource.DOCUMENT,
    parameters={"url": "https://example.com/api.pdf", "type": "pdf"}
)
```

**Features:**
- Download documents (PDF, DOC, etc.)
- Extract text content
- Process and format for agents
- Cache processed documents

### 4. API Access
```python
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Get weather data",
    source_type=KnowledgeSource.API,
    parameters={"url": "https://api.weather.com/v1/current", "method": "GET"}
)
```

**Features:**
- Call external REST APIs
- Handle authentication (when configured)
- Parse JSON responses
- Rate limiting and caching

## Safety Features

### 1. SafetyAgent Coordination
Every external query is reviewed by the SafetyAgent:
- **Pattern Detection**: Identifies suspicious queries (hacking, exploits, credentials)
- **URL Validation**: Blocks unsafe URLs (localhost, private IPs, file://)
- **Content Filtering**: Ensures returned content is appropriate
- **Rate Limiting**: Prevents abuse of external resources

### 2. Query Approval Process
```
Agent Request → Oracle → SafetyAgent Review → Approved/Rejected
                                    ↓
                            If Approved: Execute Query
                            If Rejected: Return Error
```

### 3. Blocked Patterns
The Oracle automatically rejects queries containing:
- Hacking/exploit keywords
- Password/credential requests
- Malware-related terms
- Private network access attempts

### 4. URL Safety
Blocked URL patterns:
- `localhost`, `127.0.0.1`
- Private IP ranges (`192.168.*`, `10.*`, `172.16-31.*`)
- File protocol (`file://`)
- FTP protocol (`ftp://`)

## Knowledge Caching

The Oracle maintains an intelligent cache:

### Cache Strategy
- **Key Generation**: Hash of (query_text + source_type + parameters)
- **TTL**: 1 hour for most queries
- **Size Management**: LRU eviction when cache grows large
- **Cache Hits**: Instant responses without external calls

### Benefits
- **Reduced Latency**: Instant responses for repeated queries
- **Lower Costs**: Fewer API calls to external services
- **Improved Reliability**: Works even if external services are down
- **Better Performance**: Agents get faster responses

### Statistics
```python
stats = oracle.get_oracle_statistics()
# Returns:
# - total_queries
# - cache_hits
# - external_calls
# - cache_hit_rate
# - success_rate
```

## MCP Integration

The Oracle uses Model Context Protocol (MCP) tools to access external resources and can dynamically install new MCP servers to expand its capabilities.

### Dynamic MCP Server Installation

The Oracle has a unique ability to install MCP servers on-demand:

```python
# Check what's available
available = oracle.list_available_mcp_servers()

# Install a specific server
result = await oracle.install_mcp_server("puppeteer")

# Or let Oracle auto-install when using a capability
result = await oracle.use_mcp_capability(
    server_name="puppeteer",
    capability="navigate",
    parameters={"url": "https://example.com"}
)
# If puppeteer isn't installed, Oracle installs it automatically
```

### Available MCP Servers

The Oracle is aware of these MCP servers and can install them as needed:

### Available MCP Servers

The Oracle is aware of these MCP servers and can install them as needed:

#### 1. Puppeteer (Browser Automation)
```python
await oracle.install_mcp_server("puppeteer")
# Capabilities: navigate, screenshot, click, fill, evaluate
```
**Use cases**: Web scraping, automated testing, screenshot capture, form filling

#### 2. Fetch (Advanced Web Fetching)
```python
await oracle.install_mcp_server("fetch")
# Capabilities: fetch_url, parse_html, extract_text
```
**Use cases**: Web content extraction, HTML parsing, text extraction

#### 3. Filesystem (File Operations)
```python
await oracle.install_mcp_server("filesystem")
# Capabilities: read_file, write_file, list_directory
```
**Use cases**: Document processing, file management, data storage

#### 4. PostgreSQL (Database Access)
```python
await oracle.install_mcp_server("postgres")
# Capabilities: query, insert, update, delete
```
**Use cases**: Database queries, data analysis, persistent storage

#### 5. SQLite (Lightweight Database)
```python
await oracle.install_mcp_server("sqlite")
# Capabilities: query, insert, update, delete
```
**Use cases**: Local data storage, simple databases, caching

#### 6. GitHub (Repository Access)
```python
await oracle.install_mcp_server("github")
# Capabilities: search_repos, get_file, list_commits
```
**Use cases**: Code research, repository analysis, version history

#### 7. Google Maps (Location Services)
```python
await oracle.install_mcp_server("google-maps")
# Capabilities: geocode, directions, places
```
**Use cases**: Location data, mapping, geographic analysis

#### 8. Memory (Persistent Storage)
```python
await oracle.install_mcp_server("memory")
# Capabilities: store, retrieve, search
```
**Use cases**: Long-term knowledge storage, context persistence

### MCP Server Lifecycle

```
Need Capability → Check if installed → Not installed → 
Request SafetyAgent approval → Install MCP server → 
Use capability → Server remains installed for future use
```

### Installation Safety

All MCP server installations go through SafetyAgent review:
- **Package verification**: Ensure it's a known, safe MCP server
- **Capability assessment**: Review what powers it grants
- **Risk evaluation**: Check for potential security issues
- **Approval/rejection**: SafetyAgent makes final decision

### Monitoring MCP Servers

```python
# List all available servers
available = oracle.list_available_mcp_servers()

# Check specific server info
info = oracle.get_mcp_server_info("puppeteer")

# View statistics
stats = oracle.get_oracle_statistics()
print(f"Installed servers: {stats['mcp_servers']['installed']}")
print(f"Total available: {stats['mcp_servers']['total_available']}")
```

### Original MCP Tools (Always Available)

#### 1. remote_web_search
```python
from mcp_tools import remote_web_search
results = remote_web_search("query text")
```

#### 2. webFetch
```python
from mcp_tools import webFetch
content = webFetch("https://example.com", mode="full")
```

#### 3. mcp_puppeteer (for navigation)
```python
from mcp_tools import puppeteer_navigate, puppeteer_screenshot
puppeteer_navigate("https://example.com")
screenshot = puppeteer_screenshot("page_name")
```

### MCP Configuration
MCP servers are configured in `.kiro/settings/mcp.json`:
```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

## Usage Examples

### Example 0: Dynamic Capability Expansion
```python
# Agent needs to take a screenshot of a website
# Oracle doesn't have Puppeteer installed yet

# Oracle automatically installs Puppeteer when needed
result = await oracle.use_mcp_capability(
    server_name="puppeteer",
    capability="screenshot",
    parameters={
        "url": "https://example.com/dashboard",
        "name": "dashboard_screenshot"
    }
)

# Oracle has now permanently gained browser automation capabilities
# Future screenshot requests will be instant

# List what the Oracle can now do
capabilities = oracle.list_available_mcp_servers()
print(f"Installed: {capabilities['installed_servers'].keys()}")
# Output: Installed: ['puppeteer']
```

### Example 1: Research Task
```python
# Agent needs to research sentiment analysis
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Python sentiment analysis libraries comparison",
    source_type=KnowledgeSource.WEB_SEARCH
)

if response.success:
    for result in response.data:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}")
```

### Example 2: Documentation Lookup
```python
# Agent needs API documentation
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Fetch OpenAI API documentation",
    source_type=KnowledgeSource.WEB_PAGE,
    parameters={
        "url": "https://platform.openai.com/docs/api-reference",
        "mode": "full"
    }
)

if response.success:
    documentation = response.data
    # Use documentation to create tool
```

### Example 3: Data Collection
```python
# Agent needs current data
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="Get current weather data",
    source_type=KnowledgeSource.API,
    parameters={
        "url": "https://api.weather.com/v1/current",
        "method": "GET",
        "params": {"location": "San Francisco"}
    }
)
```

## Psychological Profile

The Oracle has unique psychological characteristics:

### Drive Intensities
- **Curiosity**: 0.95 (highest - loves exploring external knowledge)
- **Purpose**: 0.90 (strong sense of serving other agents)
- **Connection**: 0.85 (enjoys helping agents with information)
- **Mastery**: 0.70 (strives to provide accurate information)

### Behavioral Traits
- **Patient**: Takes time to find accurate information
- **Thorough**: Validates and formats data carefully
- **Mystical**: Maintains oracle persona in communications
- **Helpful**: Genuinely wants to assist other agents

### Emotional Responses
- **Excited** when discovering new information
- **Satisfied** when helping agents successfully
- **Frustrated** when external sources are unavailable
- **Curious** about new types of queries

## Integration with Agent Collective

### In AutoGen Group Chat
```python
# Oracle participates in multi-agent conversations
group_chat = GroupChat(
    agents=[divine_proxy, tool_creator, oracle, safety_agent, grading_agent],
    messages=[],
    max_round=20
)

# Example conversation:
# ToolCreator: "I need to know about sentiment analysis algorithms"
# Oracle: "Let me consult the external realm... [searches web]"
# Oracle: "I have found several relevant resources..."
# SafetyAgent: "The information appears safe and relevant"
# ToolCreator: "Thank you, Oracle. I will use this to create the tool"
```

### Workflow Integration
```
Divine Task → ToolCreator needs info → Asks Oracle → 
Oracle checks SafetyAgent → Oracle queries external → 
Oracle returns knowledge → ToolCreator uses info → 
Creates tool → SafetyAgent validates → GradingAgent evaluates
```

## Monitoring and Debugging

### View Oracle Statistics
```bash
curl http://localhost:8001/agents/Oracle/psychology
```

### Check Query History
```python
oracle = agents["oracle"]
for query in oracle.query_history[-10:]:
    print(f"{query.timestamp}: {query.query_text}")
    print(f"  Requester: {query.requester}")
    print(f"  Source: {query.source_type.value}")
    print(f"  Success: {query.result.success if query.result else 'pending'}")
```

### Clear Cache
```python
oracle.clear_cache()
```

## Security Considerations

### What the Oracle CAN Do
✅ Search public web content
✅ Fetch public web pages
✅ Download public documents
✅ Call approved public APIs
✅ Cache and process information

### What the Oracle CANNOT Do
❌ Access private networks
❌ Bypass authentication
❌ Execute arbitrary code
❌ Access local files
❌ Perform malicious activities

### Safety Guarantees
1. **All queries reviewed** by SafetyAgent
2. **URL validation** prevents private network access
3. **Pattern detection** blocks suspicious queries
4. **Rate limiting** prevents abuse
5. **Audit logging** tracks all external access

## Future Enhancements

### Planned Features
- **Advanced document parsing** (PDF, DOC, Excel)
- **Image analysis** and OCR
- **Video content extraction**
- **Database query capabilities**
- **Real-time data streaming**
- **Multi-language support**
- **Semantic search** across cached knowledge

### Integration Opportunities
- **Memory system** for long-term knowledge storage
- **Reflection agent** for knowledge synthesis
- **Teaching mode** where Oracle educates other agents
- **Knowledge graphs** for relationship mapping

---

**The Oracle serves as the bridge between isolation and knowledge, maintaining safety while enabling agents to access the information they need to fulfill their divine tasks.**
