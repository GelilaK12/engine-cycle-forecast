import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from scripts import data_processing

PROCESSED_PATH = "data/processed/engine_data_processed.csv"
MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_rul_model.pkl"

def train():
    """Train a RandomForestRegressor; safe fallback for CI."""
    
    # 1. Load data
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
    else:
        try:
            df = data_processing.load_raw()
        except Exception:
            print("WARNING: Using synthetic data for CI")
            X = np.random.rand(100, 21)
            y = np.random.rand(100)
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            print(f"Dummy model saved at {MODEL_PATH}")
            return model

    # 2. Split features / target
    if "RUL" in df.columns:
        X = df.drop(columns=["RUL"])
        y = df["RUL"]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # 3. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train()
