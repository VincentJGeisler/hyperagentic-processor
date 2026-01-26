#!/usr/bin/env python3
"""
Creator Interface - Divine Communication Layer

This is the interface through which humans (creators) communicate with the agent universe.
It provides both a web UI and API for sending divine messages and receiving offerings.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CreatorInterface")

# Configuration
UNIVERSE_API_URL = "http://agent_universe:8000"
DIVINE_LOGS_DIR = Path("/divine/logs")
DIVINE_HISTORY_DIR = Path("/divine/history")

# Ensure directories exist
DIVINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
DIVINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Creator Interface",
    description="Divine Communication Layer for Hyperagentic Processor",
    version="1.0.0"
)

# Models
class DivineMessage(BaseModel):
    message: str
    priority: int = 5
    context: Optional[Dict[str, Any]] = None

class DivineResponse(BaseModel):
    message_id: str
    status: str
    agent_responses: List[Dict[str, Any]]
    timestamp: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Creator connected via WebSocket. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Creator disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected creators"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")

manager = ConnectionManager()

# Message history
message_history: List[Dict[str, Any]] = []

def save_message_to_history(message_data: Dict[str, Any]):
    """Save divine message and response to history"""
    message_history.append(message_data)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = DIVINE_HISTORY_DIR / f"divine_message_{timestamp}.json"
    
    with open(history_file, 'w') as f:
        json.dump(message_data, f, indent=2)
    
    logger.info(f"Saved message to history: {history_file}")

@app.get("/")
async def root():
    """Root endpoint - serve the Creator Interface UI"""
    return FileResponse("creator_interface/static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "divine", "timestamp": datetime.now().isoformat()}

@app.get("/universe/status")
async def get_universe_status():
    """Get current status of the agent universe"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{UNIVERSE_API_URL}/universe/status", timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get universe status: {e}")
        raise HTTPException(status_code=503, detail=f"Universe unreachable: {str(e)}")

@app.get("/agents/status")
async def get_agents_status():
    """Get psychological status of all agents"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{UNIVERSE_API_URL}/agents/status", timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get agents status: {e}")
        raise HTTPException(status_code=503, detail=f"Agents unreachable: {str(e)}")

@app.post("/divine/message")
async def send_divine_message(message: DivineMessage):
    """
    Send a divine message to the agent universe.
    
    This is the primary way creators (humans) communicate with agents.
    """
    message_id = f"divine_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info(f"Creator sends divine message: {message_id}")
    logger.info(f"Message: {message.message[:100]}...")
    
    # Prepare message for agents
    agent_message = {
        "message_id": message_id,
        "message": message.message,
        "priority": message.priority,
        "context": message.context or {},
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Send to agent universe
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{UNIVERSE_API_URL}/universe/divine_task",
                json=agent_message,
                timeout=300.0  # 5 minutes for complex tasks
            )
            response.raise_for_status()
            agent_response = response.json()
        
        # Prepare divine response
        divine_response = {
            "message_id": message_id,
            "status": "completed",
            "divine_message": message.message,
            "agent_responses": agent_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to history
        save_message_to_history(divine_response)
        
        # Broadcast to connected creators
        await manager.broadcast({
            "type": "divine_response",
            "data": divine_response
        })
        
        logger.info(f"Divine message {message_id} processed successfully")
        
        return divine_response
        
    except httpx.TimeoutException:
        logger.error(f"Divine message {message_id} timed out")
        raise HTTPException(status_code=504, detail="Agent universe processing timeout")
    except Exception as e:
        logger.error(f"Failed to process divine message: {e}")
        raise HTTPException(status_code=500, detail=f"Divine communication failed: {str(e)}")

@app.get("/divine/history")
async def get_divine_history(limit: int = 50):
    """Get history of divine messages and agent responses"""
    return {
        "total_messages": len(message_history),
        "messages": message_history[-limit:] if limit > 0 else message_history
    }

@app.get("/divine/history/{message_id}")
async def get_divine_message(message_id: str):
    """Get specific divine message by ID"""
    for msg in message_history:
        if msg.get("message_id") == message_id:
            return msg
    raise HTTPException(status_code=404, detail="Divine message not found")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time divine communication.
    
    Creators can connect here to receive live updates from the agent universe.
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "message": "Divine connection established",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "divine_message":
                # Process divine message through WebSocket
                message = DivineMessage(**data.get("data", {}))
                response = await send_divine_message(message)
                
                await websocket.send_json({
                    "type": "divine_response",
                    "data": response
                })
            
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Creator disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/agents/{agent_name}/psychology")
async def get_agent_psychology(agent_name: str):
    """Get detailed psychological profile of a specific agent"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{UNIVERSE_API_URL}/agents/{agent_name}/psychology",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get agent psychology: {e}")
        raise HTTPException(status_code=503, detail=f"Agent unreachable: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Creator Interface starting up...")
    logger.info(f"Universe API URL: {UNIVERSE_API_URL}")
    logger.info("Divine communication channels open")
    
    # Try to connect to universe
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{UNIVERSE_API_URL}/", timeout=5.0)
            logger.info(f"Successfully connected to agent universe: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to agent universe on startup: {e}")
        logger.warning("Will retry when divine messages are sent")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Creator Interface shutting down...")
    logger.info("Divine communication channels closing")

if __name__ == "__main__":
    logger.info("Starting Creator Interface...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
