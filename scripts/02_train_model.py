import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scripts.utils import save_plot

PROJECT_ROOT = r"C:\Users\kewtiiiii\Desktop\Projects\engine-cycle-forecast"

processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "fd001_processed.csv")
figures_path   = os.path.join(PROJECT_ROOT, "artifacts", "figures")
metrics_path   = os.path.join(PROJECT_ROOT, "artifacts", "metrics", "rf_performance.csv")
model_path     = os.path.join(PROJECT_ROOT, "artifacts", "models", "rf_rul_model.pkl")

os.makedirs(figures_path, exist_ok=True)
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

df = pd.read_csv(processed_path)
features = [c for c in df.columns if c != "rul"]
X = df[features]
y = df["rul"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=12, cv=3,
    scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_

y_pred = best_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
pd.DataFrame([{"MSE": mse, "R2": r2}]).to_csv(metrics_path, index=False)
print(f"✅ Model metrics saved. MSE: {mse:.2f}, R2: {r2:.2f}")

joblib.dump(best_rf, model_path)
print("✅ Trained RF model saved.")

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0,max(y_test)], [0,max(y_test)], 'r--')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL")
plt.savefig(os.path.join(figures_path, "pred_vs_actual.png"), dpi=300, bbox_inches="tight")
plt.close()


importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)

importances.to_csv(os.path.join(os.path.dirname(metrics_path), "feature_importance.csv"))

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=importances.values, y=importances.index, ax=ax)
ax.set_title("Random Forest Feature Importance")
save_plot(fig, "feature_importance.png", figures_path)
print(" Feature importance saved.")


sample_engines = df["unit"].unique()[:5]  # first 5 engines
sensors_to_plot = ["sensor_1", "sensor_2", "sensor_3"]  # pick key sensors

for sensor in sensors_to_plot:
    from scripts.utils import sensor_degradation_plot
    sensor_degradation_plot(df, sensor, sample_engines, figures_path)
