FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4

# Copy model artifacts (dari ./docker_build/model/ saat build)
COPY model/ /app/model/

EXPOSE 5000

ENTRYPOINT ["mlflow", "models", "serve", \
    "-m", "/app/model", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--no-conda"]
