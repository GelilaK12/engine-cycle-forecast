from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import requests

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="RUL Predictor API")

# ----------------------------
# Model settings
# ----------------------------
MODEL_PATH = Path("artifacts/models/rf_rul_model.pkl")
MODEL_URL = "https://github.com/GelilaK12/engine-cycle-forecast/releases/download/v1.0/rf_rul_model.pkl"  
model = None  # Global variable to hold the model

# ----------------------------
# Pydantic request schema
# ----------------------------
class SensorRequest(BaseModel):
    sensor_values: list[float]

# ----------------------------
# Load model at startup
# ----------------------------
@app.on_event("startup")
def load_model_on_startup():
    global model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Downloading model from {MODEL_URL}...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")

    model = joblib.load(MODEL_PATH)
    print("Model loaded into memory.")

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(req: SensorRequest):
    if len(req.sensor_values) != 21:
        return {"error": "Expected 21 sensor values."}

    pred = model.predict([req.sensor_values])
    return {"rul_prediction": float(pred[0])}
