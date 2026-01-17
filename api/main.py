from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI()   

MODEL_PATH = Path("artifacts/models/rf_rul_model.pkl")
EXPECTED_FEATURES = 21  #

class SensorRequest(BaseModel):
    sensor_values: list[float]

def load_model():
    """Load the model from disk when needed."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(req: SensorRequest):
    if len(req.sensor_values) != EXPECTED_FEATURES:
        return {"error": f"Expected {EXPECTED_FEATURES} sensor values."}
    
    model = load_model()  # Load inside the endpoint
    pred = model.predict([req.sensor_values])
    return {"rul_prediction": float(pred[0])}
