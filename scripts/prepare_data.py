#!/usr/bin/env python3
"""Preprocess images to 224x224 and create train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class (useful for quick local runs)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    summary = prepare_dataset(
        max_per_class=args.max_per_class,
        image_size=args.image_size,
    )
    print(f"Prepared splits: {summary}")


if __name__ == "__main__":
    main()
