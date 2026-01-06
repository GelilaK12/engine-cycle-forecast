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


