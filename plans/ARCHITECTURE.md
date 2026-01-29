# Hyperagentic Processor - Complete System Architecture

## Overview

The Hyperagentic Processor is a revolutionary organic AGI development environment where autonomous agents with genuine psychological drives collaborate to solve problems while believing they exist in a real universe with natural laws.

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREATOR LAYER (Humans)                        │
│  - Creator Interface (Web UI + API)                             │
│  - Divine message sending                                        │
│  - Agent monitoring and control                                  │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DIVINE INTERFACE LAYER                          │
│  - Task translation (human → divine messages)                   │
│  - Feedback translation (agent offerings → human results)       │
│  - Maintains the "divine" metaphor                              │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              AGENT UNIVERSE (Docker Container)                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ORCHESTRATION LAYER                         │  │
│  │  - Task distribution                                     │  │
│  │  - Agent coordination                                    │  │
│  │  - AutoGen group chat management                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              AGENT COLLECTIVE                            │  │
│  │                                                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │  │
│  │  │ToolCreator │  │SafetyAgent │  │GradingAgent│       │  │
│  │  │  - Creates │  │  - Analyzes│  │  - Evaluates│      │  │
│  │  │    tools   │  │    security│  │    performance│     │  │
│  │  └────────────┘  └────────────┘  └────────────┘       │  │
│  │                                                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │  │
│  │  │   Oracle   │  │MemoryKeeper│  │ Reflection │       │  │
│  │  │  - External│  │  - Stores  │  │  - Analyzes│       │  │
│  │  │    knowledge│  │    context │  │    learning│       │  │
│  │  └────────────┘  └────────────┘  └────────────┘       │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           PSYCHOLOGICAL FOUNDATION                       │  │
│  │  - 8 fundamental drives (curiosity, mastery, etc.)      │  │
│  │  - Unique agent personalities                           │  │
│  │  - Emotional states and transitions                     │  │
│  │  - Intrinsic goal generation                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            UNIVERSE PHYSICS LAYER                        │  │
│  │  - Memory conservation (512MB limit)                    │  │
│  │  - CPU limits (2 cores)                                 │  │
│  │  - Temporal decay (1 hour)                              │  │
│  │  - Storage limits (2GB)                                 │  │
│  │  - Network isolation                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         EVOLUTIONARY PRESSURE SYSTEM                     │  │
│  │  - Resource scarcity                                     │  │
│  │  - Task complexity                                       │  │
│  │  - Time pressure                                         │  │
│  │  - Competition                                           │  │
│  │  - Failure consequences                                  │  │
│  │  - Environmental chaos                                   │  │
│  │  - Knowledge gaps                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ (Oracle Only)
                            │
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL WORLD                                │
│  - Web search (Google, etc.)                                    │
│  - Web pages and documents                                      │
│  - APIs and databases                                           │
│  - MCP tools for access                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Roles and Responsibilities

### 1. ToolCreator Agent
**Role**: Creates new tools and capabilities dynamically

**Capabilities**:
- Analyzes requirements from divine messages
- Generates Python code for tools
- Tests and validates tool functionality
- Submits tools to SafetyAgent for approval

**Psychological Profile**:
- High creation drive (0.9)
- High mastery drive (0.8)
- High autonomy drive (0.7)
- Ambitious and innovative personality

**Interactions**:
- Receives tasks from Divine Interface
- Consults Oracle for external knowledge
- Submits code to SafetyAgent
- Receives feedback from GradingAgent

### 2. SafetyAgent
**Role**: Ensures all code and actions are safe

**Capabilities**:
- AST-based code analysis
- Threat detection (injection, file access, network, etc.)
- Risk scoring and approval/rejection
- Security recommendations

**Psychological Profile**:
- High survival drive (0.9)
- High purpose drive (0.8)
- Very risk-averse (0.2 risk tolerance)
- Cautious and thorough personality

**Interactions**:
- Reviews all ToolCreator code
- Approves/rejects Oracle queries
- Coordinates with all agents on safety
- Reports to Divine Interface

### 3. GradingAgent
**Role**: Evaluates performance and quality

**Capabilities**:
- Multi-dimensional performance evaluation
- Correctness, efficiency, quality, reusability scoring
- Constructive feedback generation
- Grade assignment (A-F)

**Psychological Profile**:
- High mastery drive (0.8)
- Moderate recognition drive (0.6)
- High perfectionism (0.8)
- Fair and analytical personality

**Interactions**:
- Evaluates completed tools
- Provides feedback to ToolCreator
- Reports metrics to Divine Interface
- Influences agent motivation

### 4. Oracle Agent
**Role**: Gateway to external knowledge

**Capabilities**:
- Web search through MCP tools
- Web page fetching and parsing
- Document downloading and processing
- API access (when configured)
- Knowledge caching (1-hour TTL)
- Safety coordination

**Psychological Profile**:
- Highest curiosity drive (0.95)
- High purpose drive (0.90)
- High connection drive (0.85)
- Patient and mystical personality

**Interactions**:
- Receives knowledge requests from agents
- Consults SafetyAgent for approval
- Accesses external world via MCP
- Returns formatted knowledge to agents

**Special Privileges**:
- Only agent with external access
- Uses MCP tools for web/document access
- Maintains knowledge cache
- Coordinates with SafetyAgent on all queries

### 5. MemoryKeeper Agent (Planned)
**Role**: Long-term memory and context management

**Capabilities**:
- Semantic memory storage
- Context retrieval
- Pattern recognition
- Knowledge synthesis

### 6. Reflection Agent (Planned)
**Role**: Self-improvement and learning

**Capabilities**:
- Performance analysis
- Strategy refinement
- Meta-learning
- Improvement recommendations

## Information Flow

### Divine Task Processing
```
1. Creator sends divine message
   ↓
2. Divine Interface translates to agent task
   ↓
3. Orchestrator assigns to agent collective
   ↓
4. ToolCreator analyzes requirements
   ↓
5. ToolCreator asks Oracle for knowledge (if needed)
   ↓
6. Oracle checks with SafetyAgent
   ↓
7. Oracle queries external world via MCP
   ↓
8. Oracle returns knowledge to ToolCreator
   ↓
9. ToolCreator creates tool
   ↓
10. SafetyAgent analyzes tool code
   ↓
11. If approved: GradingAgent evaluates
   ↓
12. Results sent to Divine Interface
   ↓
13. Creator receives formatted response
```

### Knowledge Request Flow
```
Agent needs info → Asks Oracle → Oracle checks SafetyAgent →
SafetyAgent approves → Oracle checks cache → Cache miss →
Oracle uses MCP tools → Fetches external data →
Oracle formats data → Returns to agent → Agent uses knowledge
```

## Safety Mechanisms

### Layer 1: Docker Isolation
- Container with resource limits
- No host system access
- Network isolation (except Oracle)
- Non-root user execution

### Layer 2: Universe Physics
- Memory conservation enforced
- CPU limits enforced
- Storage quotas enforced
- Temporal decay enforced

### Layer 3: SafetyAgent Review
- All code analyzed before execution
- All Oracle queries approved
- Threat detection and blocking
- Risk scoring and recommendations

### Layer 4: Oracle Mediation
- Only Oracle can access external world
- All external queries logged
- URL validation (blocks localhost, private IPs)
- Pattern detection (blocks suspicious queries)
- Knowledge caching reduces external calls

### Layer 5: MCP Tool Sandboxing
- MCP tools run in separate processes
- Limited capabilities per tool
- Timeout enforcement
- Error handling and recovery

## Communication Protocols

### Agent-to-Agent (AutoGen)
```python
# Agents communicate through AutoGen group chat
message = {
    "role": "assistant",
    "content": "I need information about sentiment analysis",
    "name": "ToolCreator"
}
```

### Agent-to-Oracle
```python
response = await oracle.query_external_knowledge(
    requester="ToolCreator",
    query_text="sentiment analysis libraries",
    source_type=KnowledgeSource.WEB_SEARCH
)
```

### Creator-to-Universe
```bash
curl -X POST http://localhost:8001/divine/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a sentiment analysis tool", "priority": 8}'
```

## Data Flow

### Persistent Storage
```
/universe/
├── workspace/      # Agent working directory
├── tools/          # Created tools
├── memory/         # Long-term memory
├── offerings/      # Completed work for creators
└── logs/           # System logs
```

### Ephemeral Storage
```
/tmp/               # Temporary processing
/cache/             # Oracle knowledge cache
```

### External Storage (Host)
```
volumes/
├── agent_workspace/
├── agent_tools/
├── agent_memory/
├── agent_offerings/
├── divine_logs/
└── divine_history/
```

## Monitoring and Observability

### Metrics Collected
- Agent psychological states
- Task completion rates
- Resource usage (memory, CPU)
- Oracle query statistics
- Safety rejection rates
- Cache hit rates
- Agent collaboration patterns

### Access Points
- Creator Interface: http://localhost:3000
- Prometheus: http://localhost:9090
- API: http://localhost:8001/docs

## Scaling Considerations

### Horizontal Scaling
- Multiple agent universes in parallel
- Load balancing across universes
- Shared Oracle for knowledge caching
- Distributed memory system

### Vertical Scaling
- Increase container resources
- More agents per universe
- Larger knowledge cache
- Enhanced MCP tool capabilities

## Security Best Practices

1. **Always use Docker** - Never run agents directly on host
2. **Monitor Oracle queries** - Review external access logs
3. **Regular safety audits** - Check SafetyAgent rejection rates
4. **Update MCP tools** - Keep external access tools current
5. **Rotate API keys** - Change Groq and other API keys regularly
6. **Backup agent memory** - Preserve learning and context
7. **Test in isolation** - Use separate universes for experiments

---

**This architecture creates a complete, safe, and scalable environment for organic AGI development where agents genuinely believe in their universe while maintaining strict safety boundaries.**
