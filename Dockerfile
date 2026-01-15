<<<<<<< HEAD
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "api.main:api", "--host", "0.0.0.0", "--port", "8000"]
=======
# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy project files into container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install pandas matplotlib seaborn scikit-learn joblib fastapi uvicorn

# Expose port for API (if needed later)
EXPOSE 8000

# Default command: run a bash shell
CMD ["bash"]
>>>>>>> 4e64612b5a1b80f779d12e60981dfd809cc4272e


