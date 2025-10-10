from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging
from datetime import datetime

from database import DatabaseManager
from ollama_client import OllamaAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Powered ESP32 Sensor System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
ai_agent = OllamaAI()

# Pydantic models
class SensorDataRequest(BaseModel):
    agent_id: str
    sensor_value: float
    timestamp: str

class CommandResponse(BaseModel):
    command: str
    reasoning: str
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Initialize database connection"""
    await db_manager.initialize()
    logger.info("AI-Powered ESP32 System started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    await db_manager.close()
    logger.info("System shutdown complete")

@app.get("/")
async def root():
    """System status"""
    return {
        "message": "AI-Powered ESP32 Sensor System",
        "status": "active",
        "features": ["Ollama AI", "PostgreSQL Database", "Intelligent LED Control"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/sensor")
async def receive_sensor_data(data: SensorDataRequest, background_tasks: BackgroundTasks):
    """Receive sensor data from ESP32 and process with AI"""
    try:
        # Get sensor history for AI context
        history = await db_manager.get_sensor_history(data.agent_id, limit=10)
        
        # Use AI to analyze and decide
        ai_decision = await ai_agent.analyze_sensor_data(data.sensor_value, history)
        
        # Determine LED status
        led_status = ai_decision["command"] == "LED_ON"
        
        # Store sensor data with AI decision
        await db_manager.store_sensor_data(
            data.agent_id,
            data.sensor_value,
            ai_decision["reasoning"],
            led_status
        )
        
        # Store the command
        await db_manager.store_command(
            data.agent_id,
            ai_decision["command"],
            ai_decision["reasoning"]
        )
        
        logger.info(f"AI Decision for {data.agent_id}: {ai_decision['command']} - {ai_decision['reasoning']}")
        
        return {
            "status": "success",
            "ai_decision": ai_decision,
            "message": f"Distance: {data.sensor_value}cm, AI decided: {ai_decision['command']}"
        }
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/command/{agent_id}")
async def get_command(agent_id: str):
    """ESP32 calls this to get the latest AI-generated command"""
    try:
        commands = await db_manager.get_latest_commands(agent_id, limit=1)
        
        if commands:
            latest_command = commands[0]
            return {
                "command": latest_command["command"],
                "reasoning": latest_command["ai_reasoning"],
                "timestamp": latest_command["timestamp"].isoformat()
            }
        else:
            return {
                "command": "LED_OFF",
                "reasoning": "No commands available",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting command: {e}")
        return {
            "command": "LED_OFF",
            "reasoning": "Error occurred",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/history/{agent_id}")
async def get_sensor_history(agent_id: str, limit: int = 20):
    """Get sensor data history for monitoring"""
    try:
        history = await db_manager.get_sensor_history(agent_id, limit)
        commands = await db_manager.get_latest_commands(agent_id, limit)
        
        return {
            "sensor_history": history,
            "command_history": commands,
            "agent_id": agent_id
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/analyze")
async def manual_ai_analysis(sensor_value: float, agent_id: str = "ESP32_001"):
    """Manually trigger AI analysis for testing"""
    try:
        history = await db_manager.get_sensor_history(agent_id, limit=10)
        ai_decision = await ai_agent.analyze_sensor_data(sensor_value, history)
        
        return {
            "sensor_value": sensor_value,
            "ai_decision": ai_decision,
            "context": f"Analyzed {len(history)} previous readings"
        }
        
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
