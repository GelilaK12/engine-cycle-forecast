from fastapi import FastAPI
from pydantic import BaseModel
import joblib

api = FastAPI(title="RUL Predictor")

model = joblib.load("rf_rul_model.pkl")

class PredictRequest(BaseModel):
    sensor_values: list[float]  

@api.post("/predict")
def predict(req: PredictRequest):
    if len(req.sensor_values) != 21:
        return {"error": "Expected 21 sensor values."}
    pred = model.predict([req.sensor_values])
    return {"rul_prediction": float(pred[0])}
