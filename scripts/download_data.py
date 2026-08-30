#!/usr/bin/env python3
"""Download Cats vs Dogs dataset (or create synthetic data for smoke runs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download import ensure_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create a tiny synthetic dataset for smoke/CI runs",
    )
    parser.add_argument("--n-per-class", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/extract even if PetImages already exists",
    )
    args = parser.parse_args()
    ensure_dataset(
        synthetic=args.synthetic,
        n_per_class=args.n_per_class,
        force=args.force,
    )


if __name__ == "__main__":
    main()
