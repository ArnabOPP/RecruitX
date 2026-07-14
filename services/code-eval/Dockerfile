FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODE_EVAL_ENVIRONMENT=production

# curl: HEALTHCHECK below. docker.io: provides the `docker` CLI binary this
# service shells out to (sandbox/docker_runner.py) — it talks to the HOST's
# Docker daemon via a mounted socket (see the docker-socket note below), it
# does not run its own nested daemon.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Non-root user — same principle as the other four services. NOTE: unlike
# those services, this container still needs write access to the mounted
# Docker socket (below) to spin up sandbox containers, which is a real,
# meaningful trust boundary despite running as non-root — see README.
RUN groupadd --gid 1000 codeeval \
    && useradd --uid 1000 --gid codeeval --create-home --home-dir /home/codeeval codeeval \
    && chown -R codeeval:codeeval /srv
USER codeeval

EXPOSE 8000

# Hits /health/ready, which only turns healthy once Docker is confirmed
# reachable and both language images are confirmed present.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# IMPORTANT — this container must be run with the host's Docker socket
# mounted in (Docker-outside-of-Docker), e.g.:
#   docker run -v /var/run/docker.sock:/var/run/docker.sock ...
# so `docker run` calls made by sandbox/docker_runner.py reach a real
# daemon that can create sibling sandbox containers. This grants this
# container real control over the host's Docker daemon — a materially
# different trust level than the other Recruitix services, none of which
# need Docker access at all. In a real deployment this service should run
# on its own dedicated host, isolated from anything else that container
# escape could reach, not co-located with other production services.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
