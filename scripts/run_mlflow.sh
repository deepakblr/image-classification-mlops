#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
exec python scripts/run_mlflow.py
