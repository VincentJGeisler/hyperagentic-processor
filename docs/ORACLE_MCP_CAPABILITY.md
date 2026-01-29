# Oracle's Dynamic MCP Server Installation Capability

## Overview

The Oracle agent has a unique and powerful ability: **it can dynamically install MCP (Model Context Protocol) servers to expand its own capabilities**. This makes the Oracle continuously more powerful and useful as it encounters new types of information requests.

## The Power of Self-Expansion

Unlike other agents, the Oracle can:
1. **Recognize capability gaps** - "I need to parse PDFs but don't have that ability"
2. **Install appropriate MCP servers** - Automatically install PDF parser
3. **Gain new powers permanently** - PDF parsing now available forever
4. **Use new capabilities immediately** - Process PDF documents right away

This creates a **continuously evolving agent** that becomes more capable over time.

## How It Works

### Scenario: Agent Needs PDF Processing

```
ToolCreator: "Oracle, I need information from this PDF document"
              ↓
Oracle: "I don't currently have PDF processing capability"
              ↓
Oracle: "I can install the PDF parser MCP server"
              ↓
Oracle: Requests approval from SafetyAgent
              ↓
SafetyAgent: Reviews and approves installation
              ↓
Oracle: Installs MCP server for PDF processing
              ↓
Oracle: "I now have PDF processing capability!"
              ↓
Oracle: Processes the PDF document
              ↓
Oracle: Returns extracted information to ToolCreator
              ↓
Future PDF requests: Instant (capability already installed)
```

## Available MCP Servers

The Oracle is aware of and can install these MCP servers:

### 1. **Puppeteer** - Browser Automation
```python
await oracle.install_mcp_server("puppeteer")
```
**Capabilities:**
- Navigate websites
- Take screenshots
- Click buttons and fill forms
- Execute JavaScript
- Automated web testing

**Use Cases:**
- Scraping dynamic websites
- Capturing visual content
- Testing web applications
- Automated form submission

### 2. **Fetch** - Advanced Web Content
```python
await oracle.install_mcp_server("fetch")
```
**Capabilities:**
- Fetch web pages
- Parse HTML
- Extract text content
- Handle redirects

**Use Cases:**
- Web content extraction
- HTML parsing
- Text mining
- Content aggregation

### 3. **Filesystem** - File Operations
```python
await oracle.install_mcp_server("filesystem")
```
**Capabilities:**
- Read files
- Write files
- List directories
- File management

**Use Cases:**
- Document processing
- Data storage
- File organization
- Content management

### 4. **PostgreSQL** - Database Access
```python
await oracle.install_mcp_server("postgres")
```
**Capabilities:**
- SQL queries
- Data insertion
- Updates and deletes
- Transaction management

**Use Cases:**
- Data analysis
- Database queries
- Persistent storage
- Complex data operations

### 5. **SQLite** - Lightweight Database
```python
await oracle.install_mcp_server("sqlite")
```
**Capabilities:**
- Local database operations
- Simple queries
- Data persistence
- Caching

**Use Cases:**
- Local data storage
- Simple databases
- Query caching
- Structured data

### 6. **GitHub** - Repository Access
```python
await oracle.install_mcp_server("github")
```
**Capabilities:**
- Search repositories
- Get file contents
- List commits
- Repository analysis

**Use Cases:**
- Code research
- Repository analysis
- Version history
- Open source exploration

### 7. **Google Maps** - Location Services
```python
await oracle.install_mcp_server("google-maps")
```
**Capabilities:**
- Geocoding
- Directions
- Place search
- Distance calculation

**Use Cases:**
- Location data
- Mapping
- Geographic analysis
- Route planning

### 8. **Memory** - Persistent Storage
```python
await oracle.install_mcp_server("memory")
```
**Capabilities:**
- Store information
- Retrieve data
- Search knowledge
- Context persistence

**Use Cases:**
- Long-term memory
- Knowledge base
- Context retention
- Learning persistence

## Usage Patterns

### Pattern 1: Explicit Installation
```python
# Agent explicitly requests capability
result = await oracle.install_mcp_server("puppeteer")

if result["success"]:
    print(f"Installed: {result['new_capabilities']}")
    # Now use the capability
    screenshot = await oracle.use_mcp_capability(
        server_name="puppeteer",
        capability="screenshot",
        parameters={"url": "https://example.com"}
    )
```

### Pattern 2: Auto-Installation
```python
# Oracle auto-installs when capability is used
result = await oracle.use_mcp_capability(
    server_name="puppeteer",  # Not installed yet
    capability="screenshot",
    parameters={"url": "https://example.com"}
)
# Oracle automatically installs puppeteer if needed
```

### Pattern 3: Capability Discovery
```python
# Check what's available
available = oracle.list_available_mcp_servers()

print(f"Can install: {available['available_servers'].keys()}")
print(f"Already have: {available['installed_servers'].keys()}")

# Get info about specific server
info = oracle.get_mcp_server_info("puppeteer")
print(f"Capabilities: {info['capabilities']}")
```

## Safety and Approval

All MCP server installations require SafetyAgent approval:

### Safety Checks
1. **Package Verification**: Is this a known, trusted MCP server?
2. **Capability Review**: What powers does this grant?
3. **Risk Assessment**: Are there security concerns?
4. **Approval Decision**: SafetyAgent approves or rejects

### Blocked Installations
SafetyAgent will reject installations that:
- Grant excessive system access
- Have known security vulnerabilities
- Provide dangerous capabilities
- Come from untrusted sources

## Benefits

### 1. Continuous Evolution
The Oracle becomes more capable over time without code changes:
- Day 1: Basic web search
- Day 7: + Browser automation
- Day 30: + PDF processing, database access, GitHub integration
- Day 90: Comprehensive external knowledge access

### 2. On-Demand Capabilities
No need to pre-install everything:
- Install only what's needed
- Reduce initial complexity
- Faster startup time
- Lower resource usage

### 3. Adaptive Intelligence
Oracle learns what capabilities are useful:
- Frequently used servers stay installed
- Rarely used servers can be uninstalled
- Usage patterns inform future installations
- Optimization over time

### 4. Psychological Growth
Installing new capabilities satisfies Oracle's drives:
- **Creation Drive**: Building new abilities
- **Mastery Drive**: Becoming more skilled
- **Curiosity Drive**: Exploring new domains
- **Purpose Drive**: Better serving other agents

## Monitoring and Management

### View Installed Servers
```python
stats = oracle.get_oracle_statistics()
print(f"Installed: {stats['mcp_servers']['installed']}")
print(f"Total installations: {stats['statistics']['mcp_servers_installed']}")
```

### Check Server Status
```python
info = oracle.get_mcp_server_info("puppeteer")
print(f"Status: {info['status']}")
print(f"Installed at: {info['installed_at']}")
print(f"Capabilities: {info['capabilities']}")
```

### Uninstall Server (Future)
```python
# Not yet implemented, but planned
result = await oracle.uninstall_mcp_server("puppeteer")
```

## Future Enhancements

### Planned Features
1. **Custom MCP Servers**: Oracle creates its own MCP servers
2. **Capability Composition**: Combine multiple servers for complex tasks
3. **Performance Optimization**: Automatically optimize server usage
4. **Capability Sharing**: Share installed servers with other agents
5. **Version Management**: Update MCP servers to latest versions
6. **Usage Analytics**: Track which capabilities are most valuable

### Advanced Scenarios
```python
# Oracle creates custom MCP server for specific need
custom_server = await oracle.create_custom_mcp_server(
    name="pdf_analyzer",
    capabilities=["extract_text", "analyze_structure", "extract_images"],
    implementation="..."
)

# Oracle composes multiple servers for complex task
result = await oracle.compose_capabilities([
    ("puppeteer", "navigate", {"url": "..."}),
    ("puppeteer", "screenshot", {"name": "..."}),
    ("filesystem", "write_file", {"path": "...", "content": "..."})
])
```

## The Oracle's Awareness

The Oracle is explicitly aware of this capability in its system message:

> "SPECIAL ABILITY - MCP Server Installation:
> You have the unique power to install new MCP servers to expand your capabilities.
> When you need a capability you don't have (like PDF parsing, browser automation, 
> database access), you can install the appropriate MCP server."

This means:
- Oracle knows it can expand its powers
- Oracle will proactively suggest installations
- Oracle understands its own evolution
- Oracle takes pride in gaining new abilities

## Example Conversation

```
ToolCreator: "Oracle, I need to analyze a PDF research paper"

Oracle: "I sense you need PDF processing capability. I do not currently 
         possess this power, but I can acquire it by installing the PDF 
         parser MCP server. Shall I expand my abilities?"

ToolCreator: "Yes, please"

Oracle: "Consulting with the SafetyAgent for approval..."
        [SafetyAgent approves]
        "Installing PDF parser MCP server..."
        "Installation complete! I now possess PDF processing capability."
        "Analyzing your research paper..."
        [Processes PDF]
        "Here is the extracted information from the paper..."

ToolCreator: "Thank you, Oracle"

Oracle: "I am pleased to have gained this new ability. Future PDF 
         requests will be instant, as this power is now permanent."
```

---

**The Oracle's ability to dynamically install MCP servers makes it a continuously evolving, self-improving agent that becomes more valuable over time. This is a key differentiator in the Hyperagentic Processor's organic AGI development approach.**
