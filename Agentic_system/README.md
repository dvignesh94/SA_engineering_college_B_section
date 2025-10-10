# AI-Powered ESP32 Sensor System

An intelligent IoT project that uses ESP32, ultrasonic sensor, LED, and AI to demonstrate agentic systems.

## What It Does
- ESP32 reads distance from ultrasonic sensor
- **AI Agent (Ollama)** analyzes sensor data and makes intelligent decisions
- **AI decides**: LED ON/OFF based on context, patterns, and safety
- All data and AI decisions stored in PostgreSQL database

## Hardware Needed
- ESP32 development board
- HC-SR04 Ultrasonic sensor
- LED + 220Ω resistor
- Jumper wires
- Breadboard

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database
```bash
python setup_database.py
```

### 3. Configure ESP32
1. Open `esp32_microagent/simple_esp32.ino` in Arduino IDE
2. Change WiFi settings:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* server_ip = "192.168.1.100";  // Your computer's IP
   ```
3. Upload to ESP32

### 4. Start AI Server
```bash
python main.py
```

### 5. Test AI System
- Open browser: http://localhost:8000/docs
- Move hand near sensor
- AI analyzes distance and intelligently controls LED
- View AI decisions and reasoning in database

## Files
- `simple_esp32.ino` - ESP32 Arduino code with AI integration
- `main.py` - FastAPI server with Ollama AI
- `ollama_client.py` - AI agent for intelligent decisions
- `database.py` - PostgreSQL database manager
- `models.py` - Database models
- `setup_database.py` - Database setup script

## AI Features
- **Ollama Integration** - Uses gpt-oss:120b model for intelligent analysis
- **Context Awareness** - AI considers sensor history and patterns
- **Intelligent Decisions** - Not just simple rules, but contextual reasoning
- **Database Storage** - All AI decisions and reasoning stored
- **Fallback Logic** - Simple rules when AI is unavailable

## The AI Logic
```python
# AI analyzes sensor data with context:
ai_decision = await ai_agent.analyze_sensor_data(distance, history)

# AI considers:
# - Current distance reading
# - Historical patterns
# - Safety implications
# - Response timing
# - Context awareness

# Returns intelligent decision with reasoning
{
    "command": "LED_ON",
    "reasoning": "Object approaching rapidly, safety concern",
    "confidence": 0.9
}
```

Perfect for learning AI integration, database management, and intelligent IoT systems!
