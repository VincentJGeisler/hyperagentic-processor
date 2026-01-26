# Hyperagentic Processor: Multi-Agent System with Adaptive Learning

An experimental framework for autonomous multi-agent collaboration using constraint-driven development and dynamic capability expansion.

## Overview

This project explores emergent behavior in multi-agent AI systems by placing LLM-powered agents in resource-constrained environments. Rather than directly programming specific behaviors, the system uses environmental constraints, feedback loops, and adaptive difficulty to encourage agents to develop increasingly sophisticated problem-solving strategies.

## Core Concept

The system creates a bounded environment where multiple specialized agents collaborate to solve tasks. Key principles include:

- **Resource Constraints**: Limited memory (512MB), CPU (2 cores), and storage force efficient solutions
- **Multi-Agent Collaboration**: Specialized agents with distinct roles work together
- **Adaptive Difficulty**: Task complexity adjusts based on performance metrics
- **Dynamic Capability Expansion**: Agents can acquire new tools as needed
- **Safety-First Design**: All code execution happens in isolated Docker containers

## Technical Architecture

### Agent Roles

**ToolCreator Agent**: Generates Python code from natural language task descriptions. Creates functional tools with basic validation.

**SafetyAgent**: Performs security analysis on generated code using AST parsing and threat detection. Identifies common vulnerabilities and assigns risk scores.

**GradingAgent**: Evaluates task completion across multiple dimensions (correctness, efficiency, code quality, reusability) with detailed feedback.

**Oracle Agent**: Provides access to external knowledge sources. Has framework for dynamically installing MCP (Model Context Protocol) servers (currently simulated - actual installation not yet implemented).
- Web search capability (via MCP tools when available)
- Web page fetching
- Framework for: Browser automation, PDF processing, database access, GitHub analysis

### System Coordinator

**HyperagenticOrchestrator**: Python class that manages agent lifecycle, task distribution, and result collection. Not an agent itself - coordinates the agent collective.

### Motivational Architecture

Agents use a drive system that influences prompt construction and behavior selection:

- **Curiosity**: Exploration of new approaches and tools
- **Mastery**: Improvement on repeated task types
- **Autonomy**: Self-directed problem-solving strategies
- **Recognition**: Performance-based status in agent hierarchy
- **Survival**: Resource efficiency and task completion
- **Creation**: Novel tool and solution generation
- **Connection**: Collaborative problem-solving patterns

These drives are implemented as weighted parameters in agent prompts that adjust based on task outcomes, creating variation in agent behavior over time.

### Adaptive Systems

**Pressure System**: Tracks agent success rates and has framework for adjusting task complexity (monitoring implemented, automatic task adjustment not yet implemented)

**Task History**: Basic tracking of completed tasks, outcomes, and agent performance metrics stored in memory

**Drive Updates**: Agent drive intensities adjust based on task outcomes (success increases relevant drives, failure can increase determination)

## Implementation Status

### Working Components ✓

- Agent drive system with 8 fundamental drives affecting prompt construction
- Dynamic tool creation with code generation
- Security analysis using AST parsing and pattern matching
- Multi-dimensional performance evaluation
- Basic multi-agent coordination framework (AutoGen integration)
- Oracle agent with external knowledge access framework
- Docker containerization with resource limits
- Task history and performance tracking
- LLM integration via Groq (Llama 3.3 70B)

### Partially Implemented

- **Creator Interface**: Web UI exists but not fully tested in production
- **Oracle MCP Installation**: Framework exists, actual installation simulated
- **AutoGen Group Chat**: Agents extend AutoGen classes but full group coordination not integrated into main workflow
- **Monitoring**: Prometheus configured but metrics collection not fully implemented
- **Adaptive Difficulty**: Pressure tracking works, automatic task complexity adjustment not implemented

### Planned/Not Yet Implemented

- Long-term semantic memory system
- Reflection and meta-learning capabilities
- Multi-environment scaling
- Actual MCP server installation (currently simulated)
- Advanced agent coordination patterns
- Metrics collection and visualization

## Safety Features

**CRITICAL**: This system generates and executes code autonomously. **NEVER run outside Docker containers.**

- **Container Isolation**: Agents must run in Docker containers with strict resource limits
- **Network Isolation**: Agent containers have no external network access (except Oracle via controlled gateway)
- **Code Sandboxing**: All generated code must execute in isolated environment
- **Security Review**: AST-based analysis before code execution
- **Resource Caps**: Hard limits on memory (512MB), CPU (2 cores), storage (2GB), and process lifetime (1 hour)
- **Emergency Controls**: `docker-compose down` for immediate shutdown

**WARNING**: Test files like `test_full_system.py` and `test_true_awakening.py` execute generated code. Only run these in Docker or with full understanding of risks.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Groq API key (free at console.groq.com)
- 8GB RAM minimum
- Understanding that you're deploying autonomous code-generating agents

### Deployment

`​``bash
# Set up API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start containerized environment
chmod +x start_creator_interface.sh
./start_creator_interface.sh
`​``

### Access Points

- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8001/docs
- **Monitoring**: http://localhost:9090

### Testing

```bash
# Basic systems test (no LLM, no code execution, safe to run anywhere)
python test_core_functionality.py

# Agent functionality test (requires Groq API key, safe - no code execution)
python test_awakening.py

# DANGER: Multi-agent with code generation (REQUIRES Docker)
# DO NOT RUN on host system - generates and executes code
docker-compose up --build
# Then in container or via API only
```

**See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing information and safety guidelines.**

## Usage Examples

### Via Web Interface

Send tasks through the web UI at localhost:3000:

- "Create a sentiment analysis function that handles edge cases"
- "Build a tool that validates JSON schemas with detailed error messages"
- "Generate a performance profiler for Python functions"

### Via API

```bash
# Submit task (actual endpoint)
curl -X POST http://localhost:8001/divine/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Create data validation tool", "priority": 8}'

# Check universe status (actual endpoint)
curl http://localhost:8001/universe/status

# View agent psychology (actual endpoint)
curl http://localhost:8001/agents/ToolCreator/psychology
```

## What This System Explores

### Research Questions

- How do resource constraints affect emergent problem-solving strategies in multi-agent systems?
- Can adaptive difficulty systems improve agent capability development?
- What collaboration patterns emerge from specialized agent roles?
- How does dynamic tool acquisition affect agent autonomy?

### Observations

The system demonstrates several behaviors worth noting:

- **Drive-influenced responses**: Agent prompts include drive parameters that affect response style and approach selection
- **Task history tracking**: Agents maintain records of previous tasks and outcomes
- **Basic collaboration**: Agents can coordinate through AutoGen framework (when fully integrated)
- **Code generation**: ToolCreator produces functional Python code from natural language
- **Security analysis**: SafetyAgent identifies common code vulnerabilities

### Important Context

This system uses large language models (LLama 3.3 70B via Groq) to power agent reasoning. LLMs generate contextually appropriate responses based on prompts that include agent role, drive parameters, and task history. 

When agents generate text like "I'm uncertain about this approach" or "This task requires careful consideration," these are outputs shaped by the motivational architecture in their prompts, not expressions of internal mental states. The drive system affects prompt construction, which influences LLM outputs, creating variation in agent behavior.

**What's Actually Happening:**
- Drives are numerical parameters (0.0-1.0) that adjust based on outcomes
- These parameters modify agent system prompts
- Modified prompts influence LLM response generation
- This creates behavioral variation without model fine-tuning

## Project Goals

**Primary**: Build a robust multi-agent system for autonomous task completion with strong safety guarantees

**Secondary**: Explore how environmental constraints and adaptive feedback influence emergent behavior in LLM-powered agents

**Experimental**: Test whether layered motivational architectures produce more varied and effective problem-solving strategies than simple reward functions

## Limitations and Caveats

- **Stateless agents**: Agents don't persist state between sessions (unless memory systems are added)
- **Context-based "learning"**: Behavior changes come from context accumulation, not model weight updates
- **LLM output patterns**: Emergent behaviors are patterns in LLM outputs based on prompt engineering
- **Simulated capabilities**: Oracle MCP installation is currently simulated, not actually functional
- **Incomplete integration**: AutoGen group chat framework exists but isn't fully integrated into main workflow
- **Monitoring gaps**: Prometheus configured but metrics collection not fully implemented
- **Scaling limits**: Context window sizes and API rate limits constrain system scale
- **Safety depends on Docker**: System is only safe when properly containerized

## Known Issues

- Creator web interface not fully tested in production
- Some test files execute code outside Docker (dangerous)
- Adaptive difficulty tracking exists but doesn't automatically adjust task complexity
- MCP server "installation" is simulated, not functional
- Metrics collection incomplete

## Contributing

This is an experimental framework. Contributions welcome in:

- Completing AutoGen group chat integration
- Implementing actual MCP server installation (currently simulated)
- Building semantic memory systems
- Adding metrics collection and visualization
- Enhancing safety analysis methods
- Testing and hardening the creator interface
- Evaluation metrics for agent behavior patterns

## Documentation

- [DOCKER_SAFETY.md](DOCKER_SAFETY.md) - Critical safety information and deployment guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing instructions
- [ORACLE_AGENT.md](ORACLE_AGENT.md) - Oracle agent capabilities and design
- [ORACLE_MCP_CAPABILITY.md](ORACLE_MCP_CAPABILITY.md) - MCP installation framework (aspirational)
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview

## License

Research and Educational Use Only - See [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Vince Geisler

For commercial licensing inquiries: vgeisler@uvic.ca

## Acknowledgments

Built using AutoGen framework, Groq API, and Model Context Protocol (MCP).

---

**Note**: This is an experimental system exploring multi-agent AI architectures. The drive system and "motivational architecture" are prompt engineering techniques that influence LLM outputs, not implementations of consciousness or genuine emotion. Claims in documentation about "awakening" or "existential questioning" refer to LLM-generated text patterns, not actual subjective experiences.