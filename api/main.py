from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import requests
from contextlib import asynccontextmanager

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
    sensor_values: list[float]  # Expecting 21 sensor values

# ----------------------------
# Lifespan handler for startup/shutdown
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Ensure model directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Download model if it doesn't exist
    if not MODEL_PATH.exists():
        print(f"Downloading model from {MODEL_URL}...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")

    # Load model
    model = joblib.load(MODEL_PATH)
    print("Model loaded into memory.")

    yield  # Application runs here
    # You can add shutdown code after this if needed

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="RUL Predictor API", lifespan=lifespan)

# ----------------------------
# Prediction endpoint with padding
# ----------------------------
@app.post("/predict")
def predict(req: SensorRequest):
    if len(req.sensor_values) != 21:
        return {"error": f"Expected 21 sensor values, got {len(req.sensor_values)}."}

    # Pad input to match trained model (26 features)
    full_features = req.sensor_values + [0] * (26 - len(req.sensor_values))

    # Debug logging
    print(f"Debug: input length = {len(full_features)}")
    print(f"Debug: input values = {full_features}")

    try:
        pred = model.predict([full_features])
        return {"rul_prediction": float(pred[0])}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

# ----------------------------
# Health check endpoint
# ----------------------------
@app.get("/health")
def health_check():
    if model is None:
        return {"status": "model not loaded"}
    return {"status": "ok", "model_features_expected": 26}
