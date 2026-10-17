# Volunteer-training coordinator (control plane only: no torch, no weights).  Build from the repository root:
#   docker build -f coordinator/Dockerfile -t drafter-coordinator .
# python:3.12-slim, pinned by digest (update deliberately: docker buildx imagetools inspect python:3.12-slim)
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# cryptography: only for DRAFTER_STORAGE=gcs (V4 signed URLs need the service account's RSA key)
RUN pip install --no-cache-dir "cryptography>=42,<47" \
 && useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin drafter \
 && mkdir -p /data && chown drafter:drafter /data

WORKDIR /app
COPY LICENSE NOTICE /app/
COPY coordinator/ /app/coordinator/

USER drafter
ENV PYTHONUNBUFFERED=1 \
    DRAFTER_DB=/data/coordinator.sqlite3 \
    DRAFTER_LOCAL_STORAGE_DIR=/data/objects \
    DRAFTER_BASE_PATH=/drafter \
    DRAFTER_PORT=8080 \
    DRAFTER_ADMIN_PORT=8081
EXPOSE 8080
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
  python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:8080' + os.environ['DRAFTER_BASE_PATH'] + '/api/v1/terms', timeout=4)"
CMD ["python", "-m", "coordinator", "serve"]
