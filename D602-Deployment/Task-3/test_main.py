# test_main.py

from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)

def test_valid_prediction():
    """Test the API with valid inputs."""
    params = {
        "arrival_airport": "PHL",   
        "local_departure_time": "1345",
        "local_arrival_time": "1659"
    }
    response = client.get("/predict/delays", params=params)
    # Check HTTP 200
    assert response.status_code == 200
    data = response.json()
    # Ensure the output exists and is a number
    assert "average_departure_delay_minutes" in data
    assert isinstance(data["average_departure_delay_minutes"], float)

def test_invalid_airport():
    #Test the API with an airport code that is not in encodings
    params = {
        "arrival_airport": "IVY",
        "local_departure_time": "1345",
        "local_arrival_time": "1659"
    }
    response = client.get("/predict/delays", params=params)
    assert response.status_code == 400
    assert "Unknown airport code" in response.text

def test_invalid_time_format():
    #Test the API with an incorrectly formatted time
    params = {
        "arrival_airport": "PHL", 
        "local_departure_time": "1:45",     
        "local_arrival_time": "1659"
    }
    response = client.get("/predict/delays", params=params)
    assert response.status_code == 400
    assert "Invalid time format" in response.text
