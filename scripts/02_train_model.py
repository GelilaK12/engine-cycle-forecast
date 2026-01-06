import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from scripts.utils import save_plot, sensor_degradation_plot

# -----------------------------
# SETTINGS
# -----------------------------
USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.init(project="cmaps_rul_rf", name="rf_rul_run")

PROJECT_ROOT = r"C:\Users\kewtiiiii\Desktop\Projects\engine-cycle-forecast"

processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "fd001_processed.csv")
figures_path   = os.path.join(PROJECT_ROOT, "artifacts", "figures")
metrics_path   = os.path.join(PROJECT_ROOT, "artifacts", "metrics", "rf_performance.csv")
model_path     = os.path.join(PROJECT_ROOT, "artifacts", "models", "rf_rul_model.pkl")

os.makedirs(figures_path, exist_ok=True)
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(processed_path)
features = [c for c in df.columns if c != "rul"]
X = df[features]
y = df["rul"]

# -----------------------------
# RANDOM FOREST + HYPERPARAMETER SEARCH
# -----------------------------
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=12, cv=3,
    scoring="neg_mean_squared_error", n_jobs=-1, random_state=42, verbose=1
)
search.fit(X, y)
best_rf = search.best_estimator_

print("✅ Best hyperparameters:", search.best_params_)

pd.DataFrame([search.best_params_]).to_csv(
    os.path.join(os.path.dirname(metrics_path), "rf_best_params.csv"), index=False
)

# -----------------------------
# K-FOLD CROSS-VALIDATION + OUTLIER DETECTION
# -----------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
mse_list, r2_list = [], []
all_preds = []

fold = 1
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    all_preds.append(pd.DataFrame({
        "unit": df.loc[X_test.index, "unit"].values,
        "actual_rul": y_test.values,
        "pred_rul": y_pred
    }))

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_list.append(mse)
    r2_list.append(r2)
    print(f"Fold {fold}: MSE={mse:.2f}, R2={r2:.2f}")

    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, y_test.max()], [0, y_test.max()], "r--")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Fold {fold}: Predicted vs Actual RUL")
    save_plot(plt.gcf(), f"pred_vs_actual_fold_{fold}.png", figures_path)
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residuals")
    plt.title(f"Fold {fold}: Residuals")
    save_plot(plt.gcf(), f"residuals_fold_{fold}.png", figures_path)
    plt.close()

    abs_errors = np.abs(residuals)
    threshold = np.percentile(abs_errors, 95)
    df_test = pd.DataFrame({
        "unit": df.loc[X_test.index, "unit"].values,
        "actual_rul": y_test.values,
        "pred_rul": y_pred,
        "abs_error": abs_errors
    })
    outliers = df_test[df_test["abs_error"] > threshold].sort_values("abs_error", ascending=False)
    outliers_path = os.path.join(os.path.dirname(metrics_path), f"rul_outliers_fold_{fold}.csv")
    os.makedirs(os.path.dirname(outliers_path), exist_ok=True)
    outliers.to_csv(outliers_path, index=False)
    print(f"✅ Fold {fold}: Saved {len(outliers)} outliers to {outliers_path}")

    fold += 1

cv_results = {
    "MSE_mean": np.mean(mse_list),
    "MSE_std": np.std(mse_list),
    "R2_mean": np.mean(r2_list),
    "R2_std": np.std(r2_list)
}
cv_metrics_path = os.path.join(os.path.dirname(metrics_path), "rf_cv_metrics.csv")
pd.DataFrame([cv_results]).to_csv(cv_metrics_path, index=False)
print(f"✅ CV metrics saved to {cv_metrics_path}")

all_preds_df = pd.concat(all_preds, ignore_index=True)
all_preds_df["abs_error"] = np.abs(all_preds_df["actual_rul"] - all_preds_df["pred_rul"])
threshold_all = np.percentile(all_preds_df["abs_error"], 95)
overall_outliers = all_preds_df[all_preds_df["abs_error"] > threshold_all].sort_values("abs_error", ascending=False)
overall_outliers_path = os.path.join(os.path.dirname(metrics_path), "rul_outliers.csv")
overall_outliers.to_csv(overall_outliers_path, index=False)
print(f"✅ Saved overall {len(overall_outliers)} top outliers to {overall_outliers_path}")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)
importances.to_csv(os.path.join(os.path.dirname(metrics_path), "feature_importance.csv"))

plt.figure(figsize=(12,6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Random Forest Feature Importance")
save_plot(plt.gcf(), "feature_importance.png", figures_path)
plt.close()

# -----------------------------
# SENSOR DEGRADATION PLOTS
# -----------------------------
sample_engines = df["unit"].unique()[:5]
sensors_to_plot = ["sensor_1", "sensor_2", "sensor_3"]
for sensor in sensors_to_plot:
    sensor_degradation_plot(df, sensor, sample_engines, figures_path)

# -----------------------------
# PARTIAL DEPENDENCE PLOTS (PDP) + EXPLAIN PREDICTIONS
# -----------------------------

features_for_pdp = [f for f in importances.index if f != "unit"]

cycle_feature = "cycle"
sensor_features = [f for f in features_for_pdp if f.startswith("sensor_")]

top_sensor_features = sensor_features[:3]
pdp_features = [cycle_feature] + top_sensor_features

fig, ax = plt.subplots(figsize=(12,6))
PartialDependenceDisplay.from_estimator(best_rf, X, features=pdp_features, ax=ax)
plt.tight_layout()
save_plot(fig, "pdp_cycle_top_sensors.png", figures_path)
plt.close()

print(f"✅ PDP saved for: {pdp_features}")

# -----------------------------
# ENGINE-LEVEL EXPLANATION PLOTS
# -----------------------------

example_engines = df["unit"].unique()[:3]

for unit in example_engines:
    df_unit = df[df["unit"] == unit]

    plt.figure(figsize=(10,4))
    for sensor in pdp_features:
        if sensor != "cycle":
            plt.plot(df_unit["cycle"], df_unit[sensor], label=sensor)

    plt.xlabel("Cycle")
    plt.ylabel("Sensor Reading")
    plt.title(f"Engine {unit}: Key Sensor Trends")
    plt.legend()
    save_plot(plt.gcf(), f"engine_{unit}_sensor_trends.png", figures_path)
    plt.close()

    y_pred_unit = best_rf.predict(df_unit[features])

    plt.figure(figsize=(8,4))
    plt.plot(df_unit["cycle"], df_unit["rul"], label="Actual RUL")
    plt.plot(df_unit["cycle"], y_pred_unit, label="Predicted RUL")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title(f"Engine {unit}: Actual vs Predicted RUL")
    plt.legend()
    save_plot(plt.gcf(), f"engine_{unit}_rul.png", figures_path)
    plt.close()


# -----------------------------
# RESIDUALS WITH CONFIDENCE INTERVAL
# -----------------------------
predictions = np.stack([tree.predict(X) for tree in best_rf.estimators_], axis=0)
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)

plt.figure(figsize=(10,6))
plt.scatter(y, best_rf.predict(X), alpha=0.5, label="Predictions")
plt.fill_between(np.arange(len(y)), lower, upper, color='orange', alpha=0.2, label="95% CI")
plt.plot([0, y.max()], [0, y.max()], "r--")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL with 95% CI")
plt.legend()
save_plot(plt.gcf(), "pred_vs_actual_with_CI.png", figures_path)
plt.close()
print("✅ Residuals with confidence intervals saved.")

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
joblib.dump(best_rf, model_path)
print("✅ Trained RF model saved.")
if USE_WANDB:
    wandb.finish()


