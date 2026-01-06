import os

PROJECT_ROOT = r"C:\Users\kewtiiiii\Desktop\Projects\engine-cycle-forecast"

PATHS = {
    "raw_data": os.path.join(PROJECT_ROOT, "data", "raw", "train_FD001.txt"),
    "processed_data": os.path.join(PROJECT_ROOT, "data", "processed", "fd001_processed.csv"),
    "figures": os.path.join(PROJECT_ROOT, "artifacts", "figures"),
    "metrics": os.path.join(PROJECT_ROOT, "artifacts", "metrics"),
    "model": os.path.join(PROJECT_ROOT, "artifacts", "models", "rf_rul_model.pkl"),
    "database": os.path.join(PROJECT_ROOT, "database", "nasa_engines.db")
}

RF_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
