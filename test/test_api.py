import pytest
from fastapi.testclient import TestClient
from api.main import app
import numpy as np

client = TestClient(app)

class DummyModel:
    def predict(self, X):
        return np.array([42.0])

@pytest.fixture(autouse=True)
def inject_dummy_model(monkeypatch):
    
    import api.main as main
    main.model = DummyModel()

def test_predict_success():
    sensor_input = [1.0] * 21  
    response = client.post("/predict", json={"sensor_values": sensor_input})
    
    assert response.status_code == 200
    data = response.json()
    assert "rul_prediction" in data
    assert data["rul_prediction"] == 42.0

def test_predict_invalid_length():
    sensor_input = [1.0] * 5  
    response = client.post("/predict", json={"sensor_values": sensor_input})
    
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"] == "Expected 21 sensor values."
