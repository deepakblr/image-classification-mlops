#!/usr/bin/env python3
"""Start the FastAPI server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "champion_cnn.pt"))


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
