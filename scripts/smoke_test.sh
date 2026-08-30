#!/usr/bin/env bash
# Thin wrapper — prefer: python scripts/smoke_test.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/smoke_test.py "${1:-http://127.0.0.1:8000}"
