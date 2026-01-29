# Oracle Integration Testing Guide

## Bugs Fixed

Fixed two critical bugs in `src/main.py`:
- **Line 318**: Changed `self.logger.info()` → `logger.info()` 
- **Line 335**: Changed `self.logger.error()` → `logger.error()`

These bugs would have caused `AttributeError` when Oracle was invoked. Now fixed.

## Step 1: Restart the System

The Docker containers need to be restarted to pick up the code changes:

```bash
cd hyperagentic-processor

# Stop the current containers
docker-compose down

# Rebuild and restart (ensures new code is loaded)
docker-compose up --build -d

# Verify containers are running
docker-compose ps
```

**Expected output**: You should see containers `hyperagentic_universe` and `creator_interface` running.

**Alternative restart** (if you prefer to see logs in foreground):
```bash
docker-compose down
docker-compose up --build
```

## Step 2: Verify System Health

Check that the system is responding:

```bash
# Test the universe endpoint
curl http://localhost:8001/universe/status

# Should return JSON with universe_id, uptime, agent_statuses, etc.
```

## Step 3: Test the Original Failing Query

Open the web interface at http://localhost:8001 (or http://localhost:3000 if configured differently).

**Test Query #1** (Original failing case):
```
ask the oracle to search the web for the origin of cats
```

**What to look for**:
- Response should complete without errors
- Result should include `oracle_knowledge` field
- Result should show `oracle` in `agents_involved` list
- No `AttributeError` in logs

## Step 4: Monitor Logs for Oracle Invocation

In a separate terminal, watch the container logs:

```bash
cd hyperagentic-processor

# Watch all container logs
docker-compose logs -f

# Or watch just the universe container
docker-compose logs -f agent_universe
```

**Key log messages to look for**:
```
INFO - Consulting Oracle for external knowledge: ask the oracle...
INFO - Coordinating collaboration for task ... with agents: ['oracle']
```

**What you should NOT see**:
- `AttributeError: 'HyperagenticOrchestrator' object has no attribute 'logger'`
- Any Python exceptions related to logger

## Step 5: Additional Test Cases

Test these queries to verify keyword detection works properly:

**Test Query #2** (Web search keyword):
```
search the web for Python tutorials
```
Expected: Oracle should be invoked

**Test Query #3** (Find keyword):
```
find information about machine learning
```
Expected: Oracle should be invoked

**Test Query #4** (Research keyword):
```
research the history of computing
```
Expected: Oracle should be invoked

**Test Query #5** (Knowledge keyword):
```
I need knowledge about quantum computing
```
Expected: Oracle should be invoked

**Test Query #6** (No Oracle keywords):
```
create a calculator tool
```
Expected: Only tool_creator should be invoked, NOT oracle

## Step 6: Verify Response Structure

Each successful Oracle query should return a response with this structure:

```json
{
  "task_id": "...",
  "status": "completed",
  "result": {
    "oracle_knowledge": {
      // The simulated/placeholder data from Oracle
    },
    "oracle_confidence": 0.8,
    "agent_contributions": {
      "oracle": {
        "knowledge": {...},
        "confidence": 0.8,
        "source": "web_search"
      }
    }
  },
  "agents_involved": ["oracle"],
  "completion_time": "..."
}
```

## Expected Behavior Notes

### About MCP Simulation
The Oracle agent currently returns **simulated/placeholder data** because:
- MCP (Model Context Protocol) servers are not actually installed
- Real web search capability is not yet functional
- This is a known limitation documented in the README

**This is expected and correct**. The integration test verifies:
✓ Oracle is correctly detected and invoked
✓ The routing logic works
✓ Oracle responds without errors
✗ NOT testing real web search (that's a separate feature)

### What Success Looks Like
- Query completes without Python exceptions
- Oracle appears in `agents_involved` list
- Response includes `oracle_knowledge` field (even if simulated)
- Logs show "Consulting Oracle for external knowledge"
- No `AttributeError` about logger

### What Failure Looks Like
- `AttributeError` about logger in logs
- Oracle not listed in `agents_involved` despite using Oracle keywords
- No `oracle_knowledge` in response
- Python exception/traceback in container logs

## Troubleshooting

### If containers won't start:
```bash
# Check for port conflicts
sudo lsof -i :8000
sudo lsof -i :8001

# Check container logs for errors
docker-compose logs
```

### If Oracle still not invoked:
1. Check that query contains Oracle keywords: oracle, search, web, find, lookup, research, information, knowledge, external, query
2. Verify code changes were actually applied to main.py
3. Confirm containers were rebuilt: `docker-compose up --build`

### If you see AttributeError:
This means the fix wasn't applied or containers weren't restarted:
1. Verify `src/main.py` lines 318 and 335 use `logger` not `self.logger`
2. Run `docker-compose down && docker-compose up --build`

## Test Report Template

After testing, provide these details:

**System Restart**: ✓ / ✗
**Health Check**: ✓ / ✗
**Original Query Test**: ✓ / ✗
**Logs Show Oracle Invocation**: ✓ / ✗
**Additional Keyword Tests**: ✓ / ✗
**Any Errors**: [None / describe]

**Response Sample**: [Paste a response JSON]

**Log Sample**: [Paste relevant log lines showing Oracle invocation]
