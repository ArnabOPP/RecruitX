FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPEECH_IO_ENVIRONMENT=production

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Non-root user — same principle as cv-parser/interview-qa.
RUN groupadd --gid 1000 speechio \
    && useradd --uid 1000 --gid speechio --create-home --home-dir /home/speechio speechio \
    && chown -R speechio:speechio /srv
USER speechio

EXPOSE 8000

# Hits /health/ready, which only turns healthy once both the STT and TTS
# providers have been confirmed reachable — catches a missing/invalid Groq
# key or an unreachable edge-tts endpoint at the container-health level.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
