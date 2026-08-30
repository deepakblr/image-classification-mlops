#!/usr/bin/env bash
# Pack Assignment 2 submission: source + configs + trained model only.
# Allowlist matches the brief — no .git, no ignore files, no .gitkeep placeholders.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-MLOps_Assignment2_submission.zip}"

REQUIRED=(
  models/champion_cnn.pt
  requirements.txt
  docker/Dockerfile
  dvc.yaml
)

for path in "${REQUIRED[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "Train first: python scripts/download_data.py && python scripts/prepare_data.py && python -m src.train" >&2
    exit 1
  fi
done

# Assignment deliverables: source code, configs (DVC/CI/CD/Docker/deploy), model artifact
INCLUDE=(
  README.md
  pyproject.toml
  requirements.txt
  requirements-api.txt
  dvc.yaml
  .dvc/config
  src
  scripts
  tests
  notebooks
  docker
  k8s
  monitoring
  .github/workflows
  models/champion_cnn.pt
)

OPTIONAL=(
  artifacts/metrics.json
  artifacts/confusion_matrix.png
  artifacts/loss_curves.png
)

manifest=()
for path in "${INCLUDE[@]}"; do
  if [[ -e "$path" ]]; then
    manifest+=("$path")
  else
    echo "Warning: expected path missing, skipping: $path" >&2
  fi
done

for path in "${OPTIONAL[@]}"; do
  if [[ -f "$path" ]]; then
    manifest+=("$path")
  fi
done

echo "=== Submission zip contents ==="
printf '  %s\n' "${manifest[@]}"
echo

rm -f "$OUT"
zip -r "$OUT" "${manifest[@]}" -x '*/__pycache__/*' -x '*.pyc'

echo
echo "Created $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "Not included: .git, .gitignore, .dockerignore, .dvcignore, .gitkeep, data/, mlruns/, .venv/"
