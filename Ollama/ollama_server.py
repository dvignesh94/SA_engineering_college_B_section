import requests
import json

# API endpoint
url = "https://ollama.com/api/chat"

# Headers
headers = {
    "Authorization": "Bearer ff02288cb9b14429ad947dce02a92ddc.EO2QCOXv1b0qQ9MN-RLFnd6B",
    "Content-Type": "application/json"
}

# Payload
payload = {
    "model": "gpt-oss:120b",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": False
}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
