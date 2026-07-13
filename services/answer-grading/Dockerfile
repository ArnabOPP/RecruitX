FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANSWER_GRADING_SEMANTIC_MODEL_NAME=all-MiniLM-L6-v2 \
    ANSWER_GRADING_ENVIRONMENT=production \
    HF_HOME=/srv/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Pre-bake the embedding model weights into the image at build time:
    # the lifespan preload otherwise hits the network for this on every
    # container's first startup, which is slow and non-deterministic — same
    # reasoning as cv-parser's transformer pre-bake.
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${ANSWER_GRADING_SEMANTIC_MODEL_NAME}')"

COPY app ./app

# Non-root user — same principle as the other three services. Needs a
# real (writable, owned) home/cache dir: HF_HOME above lives under /srv,
# chown'd to this user below, same fix cv-parser needed for the same
# reason (PermissionError on a --no-create-home user's cache dir).
RUN groupadd --gid 1000 answergrading \
    && useradd --uid 1000 --gid answergrading --create-home --home-dir /home/answergrading answergrading \
    && chown -R answergrading:answergrading /srv
USER answergrading

EXPOSE 8000

# Hits /health/ready, which only turns healthy once the embedding model has
# actually loaded and run a real inference — matters during rolling
# deploys so traffic isn't routed to a replica that would 502 every score
# request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Single worker per container: each worker loads its own copy of the
# embedding model into memory, so scale by adding replicas, not --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
