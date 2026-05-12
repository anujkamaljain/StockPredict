# ============================================================
# Stock Market ML Backend — Production Dockerfile
# Trained models are baked into the image so the API runs offline.
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building native wheels (numpy, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Bake trained models into the image so Cloud Run runs offline.
# (These must already exist locally — see cloud_training_guide.md.)
COPY data/models/ ./data/models/

# Ensure runtime data subdirs exist (caches, logs)
RUN mkdir -p data/cache data/raw data/processed data/logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fs http://localhost:8080/health || exit 1

# 2 workers fits comfortably in 2 GiB; inference runs on CPU.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 30"]
