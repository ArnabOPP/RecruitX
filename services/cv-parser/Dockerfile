FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CV_PARSER_TRANSFORMER_MODEL=dslim/bert-base-NER \
    CV_PARSER_ENVIRONMENT=production \
    HF_HOME=/srv/.cache/huggingface

# Build tooling only needed transiently for compiling wheels (e.g. torch's
# deps on some platforms); nothing here is needed at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    # Pre-bake the transformer weights into the image at build time: the
    # lifespan preload otherwise hits the network for this on every
    # container's first startup, which is slow, non-deterministic, and (see
    # below) was silently failing anyway before the cache dir existed.
    && python -c "from transformers import pipeline; pipeline('token-classification', model='${CV_PARSER_TRANSFORMER_MODEL}', aggregation_strategy='simple')"

COPY app ./app

# Run as an unprivileged user — the app never needs root, and this limits
# blast radius if a dependency (image-parsing, PDF-parsing) is ever exploited.
# Needs a real (writable, owned) home/cache dir: HF_HOME above lives under
# /srv, which is chown'd to this user below — a `--no-create-home` user
# with no writable HF cache dir made the transformer silently fail to load
# at runtime with a PermissionError.
RUN groupadd --gid 1000 cvparser \
    && useradd --uid 1000 --gid cvparser --create-home --home-dir /home/cvparser cvparser \
    && chown -R cvparser:cvparser /srv
USER cvparser

EXPOSE 8000

# Hits /health/ready (not just /health/live) so the container isn't marked
# healthy until the spaCy + transformer models have actually finished
# loading — matters during rolling deploys so traffic isn't routed to a
# replica that would otherwise 503 on every request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Single worker per container: each worker loads its own copy of the
# transformer model into memory, so scale by adding replicas (behind a load
# balancer / k8s Deployment), not by raising --workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
