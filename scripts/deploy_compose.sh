#!/usr/bin/env bash
# Deploy / update the inference stack via Docker Compose (CD target).
#
# Local (default): build from Dockerfile
#   ./scripts/deploy_compose.sh
#
# Registry mode: REGISTRY_IMAGE=ghcr.io/<owner>/cats-dogs-api:latest ./scripts/deploy_compose.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_docker.sh
source "$ROOT/scripts/ensure_docker.sh"

ensure_docker

REGISTRY_IMAGE="${REGISTRY_IMAGE:-}"

if [[ -n "$REGISTRY_IMAGE" ]]; then
  echo "Pulling registry image: $REGISTRY_IMAGE"
  docker pull "$REGISTRY_IMAGE"
  docker tag "$REGISTRY_IMAGE" cats-dogs-api:latest
  export API_IMAGE=cats-dogs-api:latest
  docker_compose -f docker/docker-compose.yml up -d --no-build
else
  if [[ ! -f models/champion_cnn.pt ]]; then
    echo "Model missing — run real training first:"
    echo "  python scripts/download_data.py && python scripts/prepare_data.py && python -m src.train"
    exit 1
  fi
  export API_IMAGE=cats-dogs-api:latest
  docker_compose -f docker/docker-compose.yml up -d --build
fi

python scripts/smoke_test.py http://127.0.0.1:8000
echo "Compose deployment healthy"
