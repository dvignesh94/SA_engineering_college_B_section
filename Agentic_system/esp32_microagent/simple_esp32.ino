#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// WiFi settings - CHANGE THESE
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Server settings - CHANGE THIS TO YOUR COMPUTER'S IP
const char* server_ip = "192.168.1.100";  // Your computer's IP address
const int server_port = 8000;
const char* agent_id = "ESP32_001";  // Unique agent ID

// Hardware pins
#define TRIG_PIN 5   // Ultrasonic sensor trigger
#define ECHO_PIN 18  // Ultrasonic sensor echo
#define LED_PIN 16   // LED

// Variables
float distance = 0;
unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 2000; // Send data every 2 seconds

void setup() {
  Serial.begin(115200);
  
  // Setup pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Read ultrasonic sensor
  distance = readDistance();
  
  // Send data to server every 2 seconds
  if (millis() - lastSend > SEND_INTERVAL) {
    sendDataToServer();
    lastSend = millis();
  }
  
  // Check for commands from server
  checkForCommands();
  
  delay(100);
}

float readDistance() {
  // Send ultrasonic pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo
  long duration = pulseIn(ECHO_PIN, HIGH);
  float distance = (duration * 0.0343) / 2; // Convert to cm
  
  return distance;
}

void sendDataToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://" + String(server_ip) + ":" + String(server_port) + "/api/sensor";
    
    // Create JSON data for AI analysis
    StaticJsonDocument<200> doc;
    doc["agent_id"] = agent_id;
    doc["sensor_value"] = distance;
    doc["timestamp"] = millis();
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Data sent to AI system successfully");
      Serial.println("AI Response: " + response);
    } else {
      Serial.println("Error sending data: " + String(httpResponseCode));
    }
    
    http.end();
  }
}

void checkForCommands() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://" + String(server_ip) + ":" + String(server_port) + "/api/command/" + String(agent_id);
    
    http.begin(url);
    int httpResponseCode = http.GET();
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      
      // Parse JSON response from AI
      StaticJsonDocument<300> doc;
      deserializeJson(doc, response);
      
      String command = doc["command"];
      String reasoning = doc["reasoning"];
      
      if (command == "LED_ON") {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("AI Decision: LED ON - " + reasoning);
      } else if (command == "LED_OFF") {
        digitalWrite(LED_PIN, LOW);
        Serial.println("AI Decision: LED OFF - " + reasoning);
      }
    }
    
    http.end();
  }
}
