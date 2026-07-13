FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INTERVIEW_QA_ENVIRONMENT=production

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Non-root user — this service has no models to pre-bake (unlike cv-parser),
# so there's no HF_HOME cache-permission gotcha here, but the principle is
# the same: never run as root.
RUN groupadd --gid 1000 interviewqa \
    && useradd --uid 1000 --gid interviewqa --create-home --home-dir /home/interviewqa interviewqa \
    && chown -R interviewqa:interviewqa /srv
USER interviewqa

EXPOSE 8000

# Hits /health/ready, which only turns healthy once the configured LLM
# provider's client actually constructed successfully (i.e. an API key is
# present) — catches a missing-credential misconfiguration at the
# container-health level, not just on the first real request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
