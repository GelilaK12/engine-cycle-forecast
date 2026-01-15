import os
import pandas as pd
import sqlite3

PROJECT_ROOT = r"C:\Users\kewtiiiii\Desktop\Projects\engine-cycle-forecast"

processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "fd001_processed.csv")
metrics_path   = os.path.join(PROJECT_ROOT, "artifacts", "metrics", "predictions_sample.csv")
db_path        = os.path.join(PROJECT_ROOT, "database", "nasa_engines.db")

os.makedirs(os.path.dirname(db_path), exist_ok=True)

df_processed = pd.read_csv(processed_path)
df_predictions = pd.read_csv(metrics_path)

conn = sqlite3.connect(db_path)

df_processed.to_sql("fd001_processed", conn, if_exists="replace", index=False)
df_predictions.to_sql("fd001_predictions", conn, if_exists="replace", index=False)

conn.close()
print("✅ Data and predictions saved to SQLite database.")
