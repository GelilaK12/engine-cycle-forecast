from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()   

model = joblib.load("artifacts/models/rf_rul_model.pkl")
EXPECTED_FEATURES = model.n_features_in_

class SensorRequest(BaseModel):
    sensor_values: list[float]

@app.post("/predict")
def predict(req: SensorRequest):
    if len(req.sensor_values) != EXPECTED_FEATURES:
        return {"error": f"Expected {EXPECTED_FEATURES} sensor values."}

    X = np.array(req.sensor_values).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"rul_prediction": float(pred)}

