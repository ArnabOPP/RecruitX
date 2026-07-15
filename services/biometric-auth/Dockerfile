FROM python:3.11.9-slim-bookworm

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIOMETRIC_AUTH_ENVIRONMENT=production \
    BIOMETRIC_AUTH_FACE_LANDMARKER_MODEL_PATH=/srv/models/face_landmarker.task \
    BIOMETRIC_AUTH_ARCFACE_MODEL_NAME=buffalo_l \
    BIOMETRIC_AUTH_DB_PATH=/srv/data/biometric.db \
    HOME=/srv

# libgl1/libglib2.0-0/libgomp1: OpenCV's imported cv2 needs these even in
# headless use (mediapipe pulls in the full opencv-python, not headless).
# libegl1/libgles2: mediapipe's FaceLandmarker specifically links against
# libGLESv2 for its GPU-capable inference graph even when actually running
# on CPU — confirmed by a real container smoke test failing at startup
# with "libGLESv2.so.2: cannot open shared object file" before this was
# added; libgl1 alone was not sufficient. Installed before the insightface
# prepare() step below, which also imports OpenCV/mediapipe-adjacent code
# at build time. curl: the Docker HEALTHCHECK below and downloading the
# MediaPipe model bundle.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-bake both vision models into the image at build time — same
# "no network hit on first request" reasoning as every other Recruitix
# service's model pre-bake (cv-parser's transformer, answer-grading's
# embedding model). MediaPipe needs the .task bundle downloaded directly;
# insightface's buffalo_l pack downloads itself to `root='~/.insightface'`
# (insightface has no env-var override — expanded via Python's HOME-based
# os.path.expanduser, not a separate insightface-specific variable) the
# first time FaceAnalysis.prepare() runs, forced here rather than left to
# the first real request. HOME=/srv (set above) is what makes this
# pre-bake actually visible at runtime: this RUN step executes as root,
# whose default HOME would be /root, but the app runs as the unprivileged
# `biometricauth` user below with its own separate home directory — without
# a shared HOME, the runtime process can't see anything root cached here
# and silently re-downloads the model from GitHub on every cold start
# instead of using the pre-bake. Confirmed by a real container smoke test:
# the model was downloading over the network at container startup before
# this fix.
RUN mkdir -p /srv/models /srv/data \
    && curl -fsSL -o /srv/models/face_landmarker.task \
       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    && python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(640, 640))"

COPY app ./app

# Non-root user — same principle as every other service. No --create-home
# (that would give this user its own /home/biometricauth, defeating the
# shared HOME=/srv the insightface pre-bake above depends on); chown make
# /srv, including the pre-baked ~/.insightface cache now living there,
# writable by this user instead.
RUN groupadd --gid 1000 biometricauth \
    && useradd --uid 1000 --gid biometricauth --no-create-home biometricauth \
    && chown -R biometricauth:biometricauth /srv
USER biometricauth

EXPOSE 8000

# Hits /health/ready, which only turns healthy once both vision models
# have actually loaded and the SQLite store is reachable — matters during
# rolling deploys so traffic isn't routed to a replica that would 500 every
# enroll/verify/liveness request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Single worker per container: each worker loads its own copy of both
# vision models into memory, so scale by adding replicas, not --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
