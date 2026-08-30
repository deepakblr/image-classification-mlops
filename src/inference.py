"""Model loading and image inference helpers."""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src import config
from src.models import build_model

_MODEL = None
_META: dict | None = None
_MODEL_PATH: Path | None = None
_TRANSFORM = None


def get_model_path() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return Path(env_path)
    return config.CHAMPION_MODEL_PATH


def _build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def load_model(force_reload: bool = False):
    global _MODEL, _META, _MODEL_PATH, _TRANSFORM
    model_path = get_model_path()

    if not force_reload and _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first: python -m src.train"
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    meta = checkpoint.get("meta", {})
    image_size = int(meta.get("image_size", config.IMAGE_SIZE))

    _MODEL = model
    _META = meta
    _MODEL_PATH = model_path
    _TRANSFORM = _build_transform(image_size)
    return _MODEL


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a model-ready batch tensor."""
    load_model()
    assert _TRANSFORM is not None
    rgb = image.convert("RGB")
    tensor = _TRANSFORM(rgb).unsqueeze(0)
    return tensor


def preprocess_bytes(data: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as img:
        return preprocess_image(img)


def predict_tensor(batch: torch.Tensor) -> dict:
    model = load_model()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    probabilities = {
        name: float(probs[i].item()) for i, name in enumerate(config.CLASS_NAMES)
    }
    return {
        "prediction": pred_idx,
        "label": config.CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_image(image: Image.Image) -> dict:
    batch = preprocess_image(image)
    return predict_tensor(batch)


def predict_bytes(data: bytes) -> dict:
    batch = preprocess_bytes(data)
    return predict_tensor(batch)
