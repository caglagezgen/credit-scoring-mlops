# Stage 1: Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies first
# This layer is cached - rebuilds are fast if requirements haven't changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
# This layer changes frequently (code changes), so keep it after dependencies
COPY app/ ./app/
COPY model/ ./model/
COPY train_model.py .
COPY Readme.md .

# Document the port (purely informational)
EXPOSE 8000

# Health check: verify container is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application
# Using 0.0.0.0 ensures it listens on all network interfaces (not just localhost)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]