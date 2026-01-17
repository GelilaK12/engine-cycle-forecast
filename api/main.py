from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI()
MODEL_PATH = Path("artifacts/models/rf_rul_model.pkl")

class SensorRequest(BaseModel):
    sensor_values: list[float]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(req: SensorRequest):
    if len(req.sensor_values) != 21:
        return {"error": "Expected 21 sensor values."}
    
    model = load_model()
    pred = model.predict([req.sensor_values])
    return {"rul_prediction": float(pred[0])}
