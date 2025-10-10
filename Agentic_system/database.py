import asyncio
import asyncpg
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # Your Neon PostgreSQL connection
        self.database_url = "postgresql://neondb_owner:npg_v0obA1IrfWja@ep-square-star-a1t0ngt2-pooler.ap-southeast-1.aws.neon.tech/sensor_data"
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def store_sensor_data(self, agent_id: str, sensor_value: float, ai_decision: str, led_status: bool):
        """Store sensor data with AI decision"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sensor_data (sensor_value, agent_id, ai_decision, led_status, reading_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, sensor_value, agent_id, ai_decision, led_status, datetime.utcnow())
                return True
        except Exception as e:
            logger.error(f"Error storing sensor data: {e}")
            return False
    
    async def store_command(self, agent_id: str, command: str, ai_reasoning: str):
        """Store AI-generated command"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_commands (agent_id, command, ai_reasoning, timestamp, status)
                    VALUES ($1, $2, $3, $4, $5)
                """, agent_id, command, ai_reasoning, datetime.utcnow(), "pending")
                return True
        except Exception as e:
            logger.error(f"Error storing command: {e}")
            return False
    
    async def get_latest_commands(self, agent_id: str, limit: int = 5) -> List[dict]:
        """Get latest commands for an agent"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT command, ai_reasoning, timestamp, status
                    FROM agent_commands 
                    WHERE agent_id = $1 
                    ORDER BY timestamp DESC 
                    LIMIT $2
                """, agent_id, limit)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting commands: {e}")
            return []
    
    async def get_sensor_history(self, agent_id: str, limit: int = 10) -> List[dict]:
        """Get sensor data history for AI context"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT sensor_value, ai_decision, led_status, reading_time
                    FROM sensor_data 
                    WHERE agent_id = $1 
                    ORDER BY reading_time DESC 
                    LIMIT $2
                """, agent_id, limit)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting sensor history: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
