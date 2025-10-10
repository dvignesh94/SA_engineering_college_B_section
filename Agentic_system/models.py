from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    reading_time = Column(DateTime, default=datetime.utcnow)
    sensor_value = Column(Float)
    agent_id = Column(String(50))
    ai_decision = Column(String(100))
    led_status = Column(Boolean, default=False)

class AgentCommands(Base):
    __tablename__ = "agent_commands"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50))
    command = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="pending")
    ai_reasoning = Column(Text)
