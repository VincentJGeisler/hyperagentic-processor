# Docker Safety and Deployment Guide

## ⚠️ Why Docker is Required

The Hyperagentic Processor creates autonomous agents that:
- **Generate and execute code** dynamically
- **Create new tools** with arbitrary functionality
- **Access system resources** (memory, CPU, storage)
- **Collaborate** to solve complex problems

**Running these agents directly on your system is dangerous.** Docker provides essential safety through:

### 1. Resource Isolation
```yaml
resources:
  limits:
    memory: 512M      # Agents can't exceed this
    cpus: '2.0'       # Maximum CPU usage
```

### 2. Network Isolation
```yaml
networks:
  agent_reality:
    internal: true    # No external internet access
```

### 3. Filesystem Isolation
- Agents can only access `/universe/` directory
- Host system is completely protected
- Generated code runs in sandbox

### 4. Process Isolation
- Agents run as non-root user
- Limited system capabilities
- Can't affect host processes

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Host System (Safe)                    │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Creator Interface (Port 3000/8001)         │ │
│  │  - Web UI for humans                               │ │
│  │  - Safe communication layer                        │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │      Agent Universe (Isolated Container)           │ │
│  │  ┌──────────────────────────────────────────────┐  │ │
│  │  │  Resource Limits:                            │  │ │
│  │  │  - 512MB RAM                                 │  │ │
│  │  │  - 2 CPU cores                               │  │ │
│  │  │  - No external network                       │  │ │
│  │  │  - Sandboxed filesystem                      │  │ │
│  │  └──────────────────────────────────────────────┘  │ │
│  │                                                      │ │
│  │  Agents:                                             │ │
│  │  - ToolCreator (generates code)                     │ │
│  │  - SafetyAgent (analyzes security)                  │ │
│  │  - GradingAgent (evaluates performance)             │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Monitoring (Port 9090)                     │ │
│  │  - Resource usage tracking                         │ │
│  │  - Agent behavior monitoring                       │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Quick Start (Safe Deployment)

### 1. Prerequisites
```bash
# Check Docker is installed
docker --version
docker-compose --version

# Ensure Docker is running
docker info
```

### 2. Configure Environment
```bash
# Create .env file with your Groq API key
echo "GROQ_API_KEY=your_actual_key_here" > .env
```

### 3. Launch System
```bash
# Make startup script executable
chmod +x start_creator_interface.sh

# Start everything
./start_creator_interface.sh
```

### 4. Verify Deployment
```bash
# Check containers are running
docker ps

# Should see:
# - hyperagentic_universe
# - creator_interface
# - universe_monitor
```

### 5. Access Interfaces
- **Creator Interface**: http://localhost:3000
- **API Docs**: http://localhost:8001/docs
- **Monitoring**: http://localhost:9090

## Container Details

### Agent Universe Container
```yaml
Service: agent_universe
Image: Built from Dockerfile
Ports: 8000 (internal API)
Resources:
  - Memory: 512MB limit
  - CPU: 2 cores limit
  - Storage: 2GB limit
Network: Isolated (no external access)
User: agent_collective (non-root)
```

### Creator Interface Container
```yaml
Service: creator_interface
Image: Built from Dockerfile.creator
Ports: 8001 (API), 3000 (Web UI)
Resources: Standard (no limits)
Network: Bridge to agent_universe
Purpose: Safe human-agent communication
```

### Monitoring Container
```yaml
Service: universe_monitor
Image: prom/prometheus
Ports: 9090 (Prometheus UI)
Purpose: Track resource usage and metrics
```

## Safety Mechanisms

### 1. Code Execution Sandboxing
All agent-generated code runs inside the container with:
- No access to host filesystem
- No network access to external services
- Limited system calls
- Resource quotas enforced

### 2. Multi-Layer Security
```
Layer 1: Docker container isolation
Layer 2: Non-root user execution
Layer 3: Resource limits (memory, CPU)
Layer 4: Network isolation
Layer 5: SafetyAgent code analysis
Layer 6: Capability restrictions
```

### 3. Emergency Controls
```bash
# Stop all agents immediately
docker-compose down

# Stop and remove all data
docker-compose down -v

# View agent logs for debugging
docker-compose logs -f agent_universe

# Restart with fresh state
docker-compose down -v && docker-compose up -d
```

## Monitoring and Debugging

### View Real-Time Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f agent_universe
docker-compose logs -f creator_interface

# Last 100 lines
docker-compose logs --tail=100 agent_universe
```

### Check Resource Usage
```bash
# Container stats
docker stats

# Detailed inspection
docker inspect hyperagentic_universe
```

### Access Prometheus Monitoring
1. Open http://localhost:9090
2. Query examples:
   - `container_memory_usage_bytes`
   - `container_cpu_usage_seconds_total`
   - `agent_task_completion_rate`

## Common Issues and Solutions

### Issue: Containers won't start
```bash
# Check Docker daemon
docker info

# Check port conflicts
lsof -i :3000
lsof -i :8000
lsof -i :8001

# View detailed logs
docker-compose logs
```

### Issue: Out of memory
```bash
# Check current usage
docker stats

# Increase Docker memory allocation
# Docker Desktop → Settings → Resources → Memory
```

### Issue: API key not working
```bash
# Verify .env file exists
cat .env

# Restart containers to reload environment
docker-compose restart
```

### Issue: Can't access web interface
```bash
# Check container is running
docker ps | grep creator_interface

# Check logs for errors
docker-compose logs creator_interface

# Verify port mapping
docker port creator_interface
```

## Development vs Production

### Development (Testing Only)
```bash
# Basic tests without Docker (SAFE - no code execution)
python test_core_functionality.py
python test_awakening.py
```

### Production (Always Use Docker)
```bash
# Full system with code execution (REQUIRES Docker)
./start_creator_interface.sh
```

## Security Best Practices

1. **Never run agents outside Docker** in production
2. **Always use .env for API keys** (never commit to git)
3. **Monitor resource usage** regularly
4. **Review generated code** before deploying to production
5. **Keep Docker images updated** for security patches
6. **Use strong API keys** and rotate regularly
7. **Limit network exposure** (use firewall rules)
8. **Regular backups** of agent memory and tools

## Cleanup

### Remove Everything
```bash
# Stop and remove containers, networks, volumes
docker-compose down -v

# Remove built images
docker rmi hyperagentic_processor_agent_universe
docker rmi hyperagentic_processor_creator_interface

# Clean up Docker system
docker system prune -a
```

### Preserve Agent Memory
```bash
# Stop but keep volumes
docker-compose down

# Restart with same state
docker-compose up -d
```

---

**Remember: The agents are powerful and autonomous. Docker containerization is not optional - it's essential for safe operation.**
