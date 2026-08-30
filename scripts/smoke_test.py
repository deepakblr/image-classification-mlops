#!/usr/bin/env python3
"""Post-deploy smoke test."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test health + predict endpoints")
    parser.add_argument(
        "base_url",
        nargs="?",
        default="http://127.0.0.1:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional JPEG/PNG to send; generates a tiny RGB image if omitted",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Smoke testing {base}")
    print("==> /health")
    health = requests.get(f"{base}/health", timeout=args.timeout)
    health.raise_for_status()
    body = health.json()
    print(json.dumps(body))
    if body.get("status") not in {"healthy", "degraded"}:
        raise SystemExit(f"Unexpected health status: {body}")
    if body.get("model_loaded") is not True:
        raise SystemExit("model_loaded must be true")

    if args.image is not None:
        image_bytes = args.image.read_bytes()
        filename = args.image.name
    else:
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(200, 120, 80)).save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        filename = "smoke.jpg"

    print("==> /predict")
    resp = requests.post(
        f"{base}/predict",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=args.timeout,
    )
    resp.raise_for_status()
    pred = resp.json()
    print(json.dumps(pred))
    if pred.get("label") not in {"cat", "dog"}:
        raise SystemExit(f"Unexpected label: {pred}")
    if "probabilities" not in pred:
        raise SystemExit("Missing probabilities in response")

    print("\nSmoke tests PASSED")


if __name__ == "__main__":
    main()
