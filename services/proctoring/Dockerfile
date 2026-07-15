FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROCTORING_ENVIRONMENT=production \
    PROCTORING_FACE_LANDMARKER_MODEL_PATH=/srv/models/face_landmarker.task

# libgl1/libglib2.0-0/libgomp1: OpenCV's imported cv2 needs these even in
# headless use (mediapipe pulls in the full opencv-python, not headless).
# libegl1/libgles2: mediapipe's FaceLandmarker specifically links against
# libGLESv2 for its GPU-capable inference graph even when actually running
# on CPU — confirmed by a real container smoke test failing at startup
# with "libGLESv2.so.2: cannot open shared object file" before this was
# added; libgl1 alone was not sufficient. curl: the Docker HEALTHCHECK
# below and downloading the MediaPipe model bundle.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-bake the MediaPipe model bundle into the image at build time — same
# "no network hit on first request" reasoning as biometric-auth's model
# pre-bake (a separate copy: each service pre-bakes its own weights rather
# than sharing a volume, keeping the two services independently
# deployable).
RUN mkdir -p /srv/models \
    && curl -fsSL -o /srv/models/face_landmarker.task \
       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

COPY app ./app

# Non-root user — same principle as every other service.
RUN groupadd --gid 1000 proctoring \
    && useradd --uid 1000 --gid proctoring --create-home --home-dir /home/proctoring proctoring \
    && chown -R proctoring:proctoring /srv
USER proctoring

EXPOSE 8000

# Hits /health/ready, which only turns healthy once the vision model has
# actually loaded and Redis is reachable — matters during rolling deploys
# so traffic isn't routed to a replica that would 500 every snapshot
# request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Single worker per container: each worker loads its own copy of the
# vision model into memory, so scale by adding replicas, not --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
