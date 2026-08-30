#!/usr/bin/env bash
# Start the full local stack: Docker Compose (API + Prometheus + Grafana) + MLflow UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_docker.sh
source "$ROOT/scripts/ensure_docker.sh"

if [[ ! -f models/champion_cnn.pt ]]; then
  echo "Missing models/champion_cnn.pt — train first: python -m src.train" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: python3.11 -m venv .venv && pip install -r requirements.txt && pip install -e ." >&2
  exit 1
fi

echo "=== Starting Docker (Colima) ==="
ensure_docker

echo "=== Starting Compose: API + Prometheus + Grafana ==="
export API_IMAGE=cats-dogs-api:latest
docker_compose -f docker/docker-compose.yml up -d --build

echo "=== Starting MLflow UI (host) ==="
mkdir -p mlruns logs
if curl -sf http://127.0.0.1:5001/ >/dev/null 2>&1; then
  echo "MLflow already running on :5001"
else
  pkill -f "mlflow ui.*5001" 2>/dev/null || true
  sleep 1
  nohup "$ROOT/.venv/bin/python" -m mlflow ui \
    --backend-store-uri "$ROOT/mlruns" \
    --host 127.0.0.1 \
    --port 5001 \
    >> logs/mlflow.log 2>&1 &
  echo $! > logs/mlflow.pid
  disown -h 2>/dev/null || true
  echo "MLflow pid $(cat logs/mlflow.pid) — log: logs/mlflow.log"
fi

echo "=== Waiting for services ==="
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null && \
     curl -sf http://127.0.0.1:9090/-/ready >/dev/null && \
     curl -sf http://127.0.0.1:3000/login >/dev/null && \
     curl -sf http://127.0.0.1:5001 >/dev/null; then
    break
  fi
  sleep 2
done

echo
echo "=== System ready ==="
echo "  API (Swagger):  http://127.0.0.1:8000/docs"
echo "  API health:     http://127.0.0.1:8000/health"
echo "  API metrics:    http://127.0.0.1:8000/metrics"
echo "  MLflow UI:      http://127.0.0.1:5001"
echo "  Prometheus:     http://127.0.0.1:9090"
echo "  Grafana:        http://127.0.0.1:3000  (admin/admin)"
echo
echo "Smoke test:"
"$ROOT/.venv/bin/python" scripts/smoke_test.py http://127.0.0.1:8000
echo
echo "Stop everything: ./scripts/stop_system.sh"
