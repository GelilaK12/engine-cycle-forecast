FROM python:3.12-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ api/
COPY scripts/ scripts/
COPY artifacts/models/rf_rul_model.pkl artifacts/models/rf_rul_model.pkl

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.main:api", "--host", "0.0.0.0", "--port", "8000"]

