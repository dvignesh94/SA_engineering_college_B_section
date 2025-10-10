import aiohttp
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OllamaAI:
    def __init__(self):
        self.base_url = "https://ollama.com/api"
        self.api_key = "ff02288cb9b14429ad947dce02a92ddc.EO2QCOXv1b0qQ9MN-RLFnd6B"
        self.model = "gpt-oss:120b"
    
    async def analyze_sensor_data(self, current_distance: float, history: List[Dict]) -> Dict:
        """Use AI to analyze sensor data and decide on LED control"""
        
        # Build context from history
        history_context = ""
        if history:
            history_context = "Recent sensor readings:\n"
            for record in history[:5]:  # Last 5 readings
                history_context += f"- Distance: {record['sensor_value']}cm, LED: {'ON' if record['led_status'] else 'OFF'}\n"
        
        # Create AI prompt
        prompt = f"""
You are an intelligent agent controlling an LED based on ultrasonic sensor distance readings.

Current situation:
- Current distance: {current_distance}cm
- Sensor type: Ultrasonic (HC-SR04)
- LED control: GPIO 16

{history_context}

Your task: Analyze the distance and decide whether to turn the LED ON or OFF.

Consider these factors:
1. Safety: Objects closer than 10cm might be dangerous
2. Patterns: Look for trends in the distance readings
3. Context: Consider the recent history of readings
4. Responsiveness: Don't flicker the LED too rapidly

Respond with ONLY a JSON object in this exact format:
{{
    "decision": "LED_ON" or "LED_OFF",
    "reasoning": "Brief explanation of your decision",
    "confidence": 0.0 to 1.0
}}

Examples:
- If distance < 5cm: LED_ON (object very close, potential danger)
- If distance 5-10cm: LED_ON (object approaching, caution needed)  
- If distance > 10cm: LED_OFF (safe distance)
- If distance is stable and > 15cm: LED_OFF (no immediate concern)
"""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
                
                async with session.post(f"{self.base_url}/chat", headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result["message"]["content"]
                        
                        # Parse AI response
                        try:
                            # Extract JSON from response
                            start = ai_response.find('{')
                            end = ai_response.rfind('}') + 1
                            json_str = ai_response[start:end]
                            
                            decision_data = json.loads(json_str)
                            
                            return {
                                "command": decision_data.get("decision", "LED_OFF"),
                                "reasoning": decision_data.get("reasoning", "AI analysis"),
                                "confidence": decision_data.get("confidence", 0.5)
                            }
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Error parsing AI response: {e}")
                            # Fallback to simple rule
                            return self._fallback_decision(current_distance)
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return self._fallback_decision(current_distance)
                        
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._fallback_decision(current_distance)
    
    def _fallback_decision(self, distance: float) -> Dict:
        """Fallback decision when AI is unavailable"""
        if distance < 10:
            return {
                "command": "LED_ON",
                "reasoning": "Object too close (fallback rule)",
                "confidence": 0.8
            }
        else:
            return {
                "command": "LED_OFF", 
                "reasoning": "Safe distance (fallback rule)",
                "confidence": 0.8
            }
