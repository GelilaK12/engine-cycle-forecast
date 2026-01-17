from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict():
    sensor_input = [1.0] * 26  # MUST match model
    response = client.post("/predict", json={"sensor_values": sensor_input})

    assert response.status_code == 200
    assert "rul_prediction" in response.json()
