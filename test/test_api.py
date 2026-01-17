from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
from sklearn.dummy import DummyRegressor

client = TestClient(app)

@patch("api.main.load_model")
def test_predict(mock_load):
    # Dummy model
    model = DummyRegressor(strategy="mean")
    model.fit([[0]*21], [0])
    mock_load.return_value = model

    sensor_input = [1.0]*21
    response = client.post("/predict", json={"sensor_values": sensor_input})

    assert response.status_code == 200
    assert "rul_prediction" in response.json()
