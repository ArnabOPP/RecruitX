FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORCHESTRATOR_ENVIRONMENT=production

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN groupadd --gid 1000 orchestrator \
    && useradd --uid 1000 --gid orchestrator --create-home --home-dir /home/orchestrator orchestrator \
    && chown -R orchestrator:orchestrator /srv
USER orchestrator

EXPOSE 8000

# Hits /health/ready, which only turns healthy once Redis (this service's
# session store) is confirmed reachable — deliberately not gated on all
# five downstream services also being up, see main.py's lifespan.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
