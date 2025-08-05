import requests
import time

# Test the logs endpoint
def test_logs():
    url = "http://127.0.0.1:8000/logs"
    data = {
        "level": "info",
        "message": "Test message from Python",
        "payload": ["test data"],
        "ts": int(time.time() * 1000)
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_logs() 