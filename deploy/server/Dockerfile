# Server trung tâm ARES — image nhẹ (~150 MB), không cần GPU.
#   docker compose -f deploy/server/docker-compose.yml up -d --build
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    ARES_SERVER_HOST=0.0.0.0 ARES_SERVER_PORT=8200 ARES_SERVER_DATA_DIR=/data
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY ares_server ./ares_server
RUN useradd -r -u 10001 ares && mkdir -p /data && chown ares /data
USER ares
VOLUME ["/data"]
EXPOSE 8200
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8200/healthz',timeout=4)"
CMD ["python", "-m", "ares_server", "run"]
