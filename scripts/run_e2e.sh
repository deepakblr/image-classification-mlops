#!/usr/bin/env bash
# End-to-end local verification.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=ensure_docker.sh
source "$ROOT/scripts/ensure_docker.sh"

export MPLCONFIGDIR="$ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

CI_SMOKE=0
MAX_PER_CLASS="${MAX_PER_CLASS:-1500}"
EPOCHS="${EPOCHS:-3}"

for arg in "$@"; do
  case "$arg" in
    --ci-smoke) CI_SMOKE=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

echo "=== 0. Docker runtime ==="
ensure_docker

source .venv/bin/activate

if [[ "$CI_SMOKE" -eq 1 ]]; then
  echo "=== 1. Python pipeline (smoke) ==="
  python -m src.train --smoke
else
  echo "=== 1. Python pipeline ==="
  if [[ ! -d data/raw/PetImages/Cat ]]; then
    python scripts/download_data.py
  fi
  python scripts/prepare_data.py --max-per-class "$MAX_PER_CLASS"
  python -m src.train --epochs "$EPOCHS"
fi

ruff check src/ tests/
pytest tests/ -q

echo "=== 2. Docker image + API smoke ==="
docker build -f docker/Dockerfile -t cats-dogs-api:latest .
docker rm -f cats-dogs-test 2>/dev/null || true
docker run -d --name cats-dogs-test -p 8000:8000 cats-dogs-api:latest
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then break; fi
  sleep 1
done
# Prefer a real sample image when available
SAMPLE=""
if [[ -d data/processed/test/cat ]]; then
  SAMPLE="$(find data/processed/test/cat -type f | head -1 || true)"
fi
if [[ -n "$SAMPLE" ]]; then
  python scripts/smoke_test.py http://127.0.0.1:8000 --image "$SAMPLE"
else
  python scripts/smoke_test.py http://127.0.0.1:8000
fi
docker rm -f cats-dogs-test

echo "=== 3. Compose + monitoring ==="
docker_compose -f docker/docker-compose.yml up -d --build
sleep 8
python scripts/smoke_test.py http://127.0.0.1:8000
curl -sf http://127.0.0.1:9090/-/ready && echo " Prometheus ready"

echo ""
echo "=== END-TO-END PIPELINE PASSED ==="
