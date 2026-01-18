# Predictive Maintenance for Turbofan Engines Using RUL Regression

## Project Overview

This project predicts the Remaining Useful Life (RUL) of turbofan engines using multivariate sensor time-series data from the NASA CMAPSS dataset. Predicting RUL is critical for proactive maintenance, reducing operational risk, and minimizing downtime.

This project demonstrates production-style ML workflow ownership,including:
* Data preprocessing, feature engineering, and RUL computation

* Model training, evaluation, and hyperparameter tuning

* Artifact persistence (models, metrics, figures) and database storage

* API deployment with FastAPI for real-time predictions

* Containerization with Docker and automated CI/CD using GitHub Actions


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

| Script                       | Purpose                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `scripts/data_processing.py` | EDA, RUL computation, CSV saving                                 |
| `scripts/train_model.py`     | Random Forest training, predictions, metrics, feature importance |
| `scripts/save_to_sql.py`     | Store processed data, predictions, and metrics in SQLite         |
| `scripts/utils.py`           | Centralized plotting and helper functions                        |
| `api/main.py`                | FastAPI deployment for RUL prediction                            |


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
* Automatic Padding: input is padded to 26 features to match trained model without breaking predictions
* Dockerized Pipeline: ensures reproducible deployment across machines
* CI/CD with GitHub Actions: automated testing and deployment of API and Docker containers
* Test Coverage: unit tests for data processing, training, and API endpoints

---
    engine-cycle-forecast/
    │
    ├── api/                # FastAPI app
    ├── artifacts/          # Saved models, metrics, figures
    ├── data/               # Raw and processed datasets
    ├── database/           # SQLite DB storage
    ├── notebooks/          # Exploratory notebooks
    ├── scripts/            # ML pipeline scripts
    ├── test/               # Unit tests
    ├── wandb/              # Experiment tracking outputs
    ├── .github/workflows/  # CI/CD GitHub Actions
    ├── Dockerfile
    ├── requirements.txt
    └── README.md

---
## How to Use

### Clone the repo:

    git clone https://github.com/GelilaK12/engine-cycle-forecast.git
    cd engine-cycle-forecast
### Install dependencies:
    pip install -r requirements.txt
### Run API locally:
    uvicorn api.main:app --reload
Swagger UI:
Open http://127.0.0.1:8000/docs to send a POST request:
     
     {
      "sensor_values": [1,2,3,...,21]
     }


### Health check endpoint:
   
    GET /health 
   to confirm the model is loaded
## Performance Summary

| Metric | Baseline RF | Optimized RF | K-Fold RF |
|--------|-------------|--------------|-----------|
| MSE    | 1282.02     | 1257.60      | 432.96    |
| R²     | 0.72        | 0.73         | 0.91      |


Random Forest successfully captures the degradation trends and provides reliable predictions across engines.

---

## Tools & Technologies

* Python
* pandas, NumPy
* scikit-learn, Random Forest, XGBoost
* matplotlib, seaborn
* SQLite
* FastAPI, Uvicorn
* Docker
* GitHub Actions (CI/CD)
* VS Code for modular scripts
