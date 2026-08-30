#!/usr/bin/env python3
"""Start the MLflow UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ROOT / "mlruns"
MLRUNS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        str(MLRUNS),
        "--host",
        "127.0.0.1",
        "--port",
        "5001",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
