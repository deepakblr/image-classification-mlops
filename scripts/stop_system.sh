#!/usr/bin/env bash
# Stop Compose stack + MLflow UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_docker.sh
source "$ROOT/scripts/ensure_docker.sh"

echo "=== Stopping Compose stack ==="
if docker info >/dev/null 2>&1; then
  docker_compose -f docker/docker-compose.yml down 2>/dev/null || true
else
  echo "Docker not running — skipping compose down"
fi

echo "=== Stopping MLflow UI ==="
if [[ -f logs/mlflow.pid ]]; then
  kill "$(cat logs/mlflow.pid)" 2>/dev/null || true
  rm -f logs/mlflow.pid
fi
pkill -f "mlflow ui.*5001" 2>/dev/null || true
pkill -f "gunicorn.*127.0.0.1:5001" 2>/dev/null || true

echo "All services stopped."
