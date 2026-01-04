# Predictive Maintenance for Turbofan Engines Using RUL Regression

## Project Overview

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using multivariate sensor time-series data from the NASA CMAPSS dataset. Predicting RUL is critical for proactive maintenance, reducing operational risk, and minimizing downtime. The goal was to build a fully reproducible, modular ML pipeline that handles messy real-world time-series data, trains robust models, and provides interpretable insights into sensor contributions.

This project demonstrates **production-style ML workflow ownership**, including data preprocessing, feature engineering, model training, evaluation, artifact persistence, and database storage.


## Problem Framing

**Objective:**
Predict the number of cycles remaining before engine failure (RUL) using sensor readings and operational settings.

**Type:** Regression (continuous target)

**Evaluation Metrics:**

* Mean Squared Error (MSE)
* R²
* Visual inspection of sensor degradation trends

**Why RUL regression matters:**
Correctly estimating RUL allows maintenance teams to schedule interventions before failure occurs. Over- or underestimating RUL can lead to unnecessary costs or catastrophic equipment failure.

---

## Data

* **Source:** NASA CMAPSS Turbofan Engine Degradation Simulation Dataset
* **Format:** Space-separated text files; each row represents a time cycle for a specific engine unit
* **Features:**

  * Operational settings (control variables)
  * 21 sensor measurements reflecting engine degradation
* **Preprocessing steps:**

  * Missing values and constant columns removed
  * RUL computed per engine
  * Data saved in processed CSV for reproducibility

---

## Modeling Approach

### Pipeline Overview

```
Raw Data → Preprocessing & EDA → RUL Computation → Train/Test Split → Model Training → Evaluation → Feature Importance → Persistence → Deployment
```

* **Scripts modularized for clarity and reproducibility:**

  * `01_data_processing.py`: EDA, RUL computation, CSV saving
  * `02_train_model.py`: Random Forest training, predictions, metrics, feature importance
  * `03_save_to_sql.py`: Store processed data, predictions, and metrics in SQLite
  * `utils.py`: Centralized plotting and helper functions

### Models Explored

* Random Forest Regressor
* XGBoost Regressor

**Observations:**

* Both models achieved similar R² (~0.72)
* Random Forest had lower MSE, making it the final choice

**Hyperparameter Optimization:**

* RandomizedSearchCV improved MSE from 1282 → 1257
* R² improved slightly (0.72 → 0.73)
* Optimization confirmed that performance gains were real without sacrificing model stability

---

## Key Insights

* **Sensor degradation plots** reveal which sensors most strongly correlate with RUL, adding interpretability beyond numerical metrics.
* Modular scripts make debugging and experimentation straightforward.
* Executing scripts as modules (e.g., `python -m scripts.03_save_to_sql`) prevents import errors and enforces package structure discipline.
* Automated saving of metrics, figures, predictions, and models converts exploratory code into a **reproducible ML pipeline**.

---

## Performance Summary

| Metric | Baseline RF | Optimized RF |
| ------ | ----------- | ------------ |
| MSE    | 1282.02     | 1257.60      |
| R²     | 0.72        | 0.73         |

Random Forest successfully captures the degradation trends and provides reliable predictions across engines.

---

## Tools & Technologies

* Python
* pandas, NumPy
* scikit-learn, Random Forest
* matplotlib, seaborn
* SQLite
* VS Code for modular scripts

---

## Next Steps / Deployment

* Containerize the pipeline with Docker for reproducibility
* Deploy model as an API using FastAPI for real-time inference
* Implement logging and monitoring to track RUL predictions over time
* Extend pipeline to handle incremental sensor data updates
