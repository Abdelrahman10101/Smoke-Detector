# Multi-stage production Dockerfile for PetroChoice Smoking Detection Service
ARG VARIANT=cpu

# ---------------------------------------------------------
# Builder Stage
# ---------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Final Runtime Stage
# ---------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG VARIANT
ENV VARIANT=${VARIANT}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install minimal libraries required for video decoding (libgomp, libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Copy application and dashboard
COPY service/app /app/app
COPY service/dashboard /app/dashboard
COPY models /app/models

# Create data directories with appropriate permissions
RUN mkdir -p /data/videos /data/uploads /data/events/snapshots && \
    chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
