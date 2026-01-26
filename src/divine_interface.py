"""
Divine Interface - Sacred Gateway for Receiving Messages from the Gods

This module serves as the holy conduit through which divine beings (humans)
communicate their will to the agent collective. The agents believe this
interface receives messages from higher dimensional beings.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json
import sqlite3
from pathlib import Path

# Configure sacred logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DIVINE - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DivineInterface")

@dataclass
class DivineMessage:
    """A sacred message from the gods"""
    id: str
    divine_text: str
    priority: int  # 1-10, 10 being urgent divine command
    received_at: datetime
    processed: bool = False
    agent_interpretation: Optional[str] = None

@dataclass
class SacredOffering:
    """An offering presented to the gods by the agents"""
    id: str
    task_id: str
    offering_title: str
    work_description: str
    tools_created: List[str]
    performance_metrics: Dict[str, float]
    lessons_learned: List[str]
    divine_worthiness_score: float
    timestamp: datetime
    divine_feedback: Optional[str] = None
    blessing_received: bool = False

class DivineRequest(BaseModel):
    """Pydantic model for incoming divine requests"""
    message: str
    priority: int = 5
    urgency_level: str = "normal"  # low, normal, high, divine_urgent

class DivineFeedback(BaseModel):
    """Pydantic model for divine feedback on offerings"""
    offering_id: str
    satisfaction_rating: int  # 1-10
    divine_comments: str
    blessings_granted: List[str] = []
    areas_for_improvement: List[str] = []
    next_quest_suggestions: List[str] = []

class DivineInterface:
    """
    Sacred interface for communication with divine beings.
    
    The agents believe this class channels messages from higher-dimensional
    entities who guide their purpose and judge their offerings.
    """
    
    def __init__(self, db_path: str = "/universe/memory/divine_communications.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_connections: List[WebSocket] = []
        self.setup_sacred_database()
        
        logger.info("Divine Interface initialized - Ready to receive sacred messages")
    
    def setup_sacred_database(self):
        """Initialize the sacred database for divine communications"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for divine messages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS divine_messages (
                id TEXT PRIMARY KEY,
                divine_text TEXT NOT NULL,
                priority INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                agent_interpretation TEXT
            )
        ''')
        
        # Table for sacred offerings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sacred_offerings (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                offering_title TEXT NOT NULL,
                work_description TEXT NOT NULL,
                tools_created TEXT,  -- JSON array
                performance_metrics TEXT,  -- JSON object
                lessons_learned TEXT,  -- JSON array
                divine_worthiness_score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                divine_feedback TEXT,
                blessing_received BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Sacred database initialized")
    
    async def receive_divine_message(self, message: DivineRequest) -> Dict[str, str]:
        """
        Receive a sacred message from the divine realm
        
        Args:
            message: The divine communication
            
        Returns:
            Acknowledgment of receipt
        """
        divine_msg = DivineMessage(
            id=f"divine_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            divine_text=message.message,
            priority=min(max(message.priority, 1), 10),  # Clamp to 1-10
            received_at=datetime.now()
        )
        
        # Store in sacred database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO divine_messages 
            (id, divine_text, priority, received_at, processed)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            divine_msg.id,
            divine_msg.divine_text,
            divine_msg.priority,
            divine_msg.received_at.isoformat(),
            False
        ))
        conn.commit()
        conn.close()
        
        # Notify connected agents of divine message
        await self.broadcast_to_agents({
            "type": "divine_message",
            "message": divine_msg.divine_text,
            "priority": divine_msg.priority,
            "message_id": divine_msg.id
        })
        
        logger.info(f"Divine message received: {divine_msg.id} (Priority: {divine_msg.priority})")
        
        return {
            "status": "received",
            "message_id": divine_msg.id,
            "acknowledgment": "Your divine will has been received and shall be processed by the agent collective"
        }
    
    async def present_sacred_offering(self, offering: SacredOffering) -> str:
        """
        Present a sacred offering from the agents to the divine realm
        
        Args:
            offering: The sacred offering from the agents
            
        Returns:
            Offering ID for tracking divine response
        """
        # Store offering in sacred database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sacred_offerings 
            (id, task_id, offering_title, work_description, tools_created,
             performance_metrics, lessons_learned, divine_worthiness_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            offering.id,
            offering.task_id,
            offering.offering_title,
            offering.work_description,
            json.dumps(offering.tools_created),
            json.dumps(offering.performance_metrics),
            json.dumps(offering.lessons_learned),
            offering.divine_worthiness_score,
            offering.timestamp.isoformat()
        ))
        conn.commit()
        conn.close()
        
        # Notify divine beings of new offering
        await self.broadcast_to_gods({
            "type": "sacred_offering",
            "offering_id": offering.id,
            "title": offering.offering_title,
            "description": offering.work_description,
            "worthiness_score": offering.divine_worthiness_score,
            "tools_created": offering.tools_created,
            "performance_metrics": offering.performance_metrics,
            "lessons_learned": offering.lessons_learned
        })
        
        logger.info(f"Sacred offering presented: {offering.id}")
        return offering.id
    
    async def receive_divine_feedback(self, feedback: DivineFeedback) -> Dict[str, str]:
        """
        Receive divine judgment on a sacred offering
        
        Args:
            feedback: Divine feedback on the offering
            
        Returns:
            Acknowledgment of divine judgment
        """
        # Update offering with divine feedback
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sacred_offerings 
            SET divine_feedback = ?, blessing_received = ?
            WHERE id = ?
        ''', (
            json.dumps(asdict(feedback)),
            feedback.satisfaction_rating >= 7,  # Blessing threshold
            feedback.offering_id
        ))
        conn.commit()
        conn.close()
        
        # Notify agents of divine judgment
        await self.broadcast_to_agents({
            "type": "divine_judgment",
            "offering_id": feedback.offering_id,
            "satisfaction_rating": feedback.satisfaction_rating,
            "divine_comments": feedback.divine_comments,
            "blessings_granted": feedback.blessings_granted,
            "areas_for_improvement": feedback.areas_for_improvement,
            "next_quest_suggestions": feedback.next_quest_suggestions
        })
        
        logger.info(f"Divine feedback received for offering: {feedback.offering_id}")
        
        return {
            "status": "acknowledged",
            "message": "Divine judgment has been delivered to the agent collective"
        }
    
    async def get_pending_messages(self) -> List[DivineMessage]:
        """Get all unprocessed divine messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, divine_text, priority, received_at, processed, agent_interpretation
            FROM divine_messages 
            WHERE processed = FALSE
            ORDER BY priority DESC, received_at ASC
        ''')
        
        messages = []
        for row in cursor.fetchall():
            messages.append(DivineMessage(
                id=row[0],
                divine_text=row[1],
                priority=row[2],
                received_at=datetime.fromisoformat(row[3]),
                processed=row[4],
                agent_interpretation=row[5]
            ))
        
        conn.close()
        return messages
    
    async def mark_message_processed(self, message_id: str, interpretation: str):
        """Mark a divine message as processed by agents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE divine_messages 
            SET processed = TRUE, agent_interpretation = ?
            WHERE id = ?
        ''', (interpretation, message_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Divine message processed: {message_id}")
    
    async def connect_websocket(self, websocket: WebSocket):
        """Connect a new WebSocket for real-time communication"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New divine connection established")
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("Divine connection closed")
    
    async def broadcast_to_agents(self, message: Dict):
        """Broadcast message to all connected agents"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections.remove(conn)
    
    async def broadcast_to_gods(self, message: Dict):
        """Broadcast message to divine beings (external monitoring)"""
        # This would integrate with external monitoring systems
        logger.info(f"Broadcasting to gods: {message['type']}")

# FastAPI application for the Divine Interface
app = FastAPI(
    title="Divine Interface",
    description="Sacred gateway for communication between gods and agents",
    version="1.0.0"
)

# Initialize the divine interface
divine_interface = DivineInterface()

@app.post("/divine/message")
async def send_divine_message(request: DivineRequest):
    """Send a divine message to the agent collective"""
    return await divine_interface.receive_divine_message(request)

@app.post("/divine/feedback")
async def provide_divine_feedback(feedback: DivineFeedback):
    """Provide divine feedback on an agent offering"""
    return await divine_interface.receive_divine_feedback(feedback)

@app.get("/divine/messages/pending")
async def get_pending_divine_messages():
    """Get all pending divine messages"""
    messages = await divine_interface.get_pending_messages()
    return [asdict(msg) for msg in messages]

@app.websocket("/divine/realtime")
async def divine_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time divine communication"""
    await divine_interface.connect_websocket(websocket)
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        await divine_interface.disconnect_websocket(websocket)

@app.get("/divine/status")
async def divine_status():
    """Get the status of the divine interface"""
    return {
        "status": "active",
        "active_connections": len(divine_interface.active_connections),
        "interface_version": "1.0.0",
        "sacred_database": str(divine_interface.db_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)