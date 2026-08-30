#!/usr/bin/env python3
"""Collect simulated production requests with true labels for post-deploy tracking."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from src import config
from src.data.preprocess import load_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.ARTIFACTS_DIR / "post_deploy_tracking.json",
    )
    args = parser.parse_args()

    splits = load_splits()
    pool = splits.get("test") or splits["val"]
    sample = random.sample(pool, k=min(args.n, len(pool)))

    rows = []
    correct = 0
    for rec in sample:
        path = config.PROJECT_ROOT / rec["path"]
        with open(path, "rb") as fh:
            resp = requests.post(
                f"{args.base_url}/predict",
                files={"file": (path.name, fh, "image/jpeg")},
                timeout=30,
            )
        resp.raise_for_status()
        body = resp.json()
        true_label = rec["class_name"]
        pred = body["label"]
        ok = pred == true_label
        correct += int(ok)
        rows.append(
            {
                "path": rec["path"],
                "true_label": true_label,
                "predicted_label": pred,
                "confidence": body["confidence"],
                "correct": ok,
            }
        )

    report = {
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"n": report["n"], "accuracy": report["accuracy"]}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
