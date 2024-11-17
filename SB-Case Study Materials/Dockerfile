# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOGGING_LEVEL=INFO \
    RANDOM_SEED=42

# Copy requirements first to leverage Docker cache
COPY ["SB-Case Study Materials/requirements.txt", "./"]

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY SB-Case\ Study\ Materials/notebooks/ /app/notebooks/

# Create volume mount points
VOLUME ["./SB-Case Study Materials/data", "./SB-Case Study Materials/models", "./SB-Case Study Materials/logs", "./SB-Case Study Materials/plots", "./SB-Case Study Materials/reports"]

# Set default command with health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "-u", "notebooks/main.py"]