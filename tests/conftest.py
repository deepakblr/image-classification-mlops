"""Shared pytest fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from src import config
from src.data.download import create_synthetic_dataset
from src.models import build_model
from src.train import save_checkpoint


@pytest.fixture()
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(180, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def synthetic_raw(tmp_path, monkeypatch) -> Path:
    raw = tmp_path / "raw"
    create_synthetic_dataset(raw, n_per_class=20, image_size=48)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    processed = tmp_path / "processed"
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "SPLITS_PATH", processed / "splits.json")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return raw


@pytest.fixture()
def trained_model(tmp_path, monkeypatch) -> Path:
    model_path = tmp_path / "champion_cnn.pt"
    model = build_model()
    save_checkpoint(
        model,
        model_path,
        meta={"smoke": True, "image_size": config.IMAGE_SIZE, "test_metrics": {}},
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import src.inference as inference

    inference._MODEL = None
    inference._MODEL_PATH = None
    inference._META = None
    inference._TRANSFORM = None
    return model_path
