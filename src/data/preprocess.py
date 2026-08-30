"""Preprocess images: resize, split train/val/test, build path manifests."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

from src import config


def list_class_images(raw_dir: Path | None = None) -> dict[str, list[Path]]:
    """Return valid image paths grouped by class name (cat/dog)."""
    raw_dir = raw_dir or config.RAW_DIR
    pet = raw_dir / "PetImages"
    mapping = {"cat": pet / "Cat", "dog": pet / "Dog"}
    result: dict[str, list[Path]] = {}
    for class_name, folder in mapping.items():
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing {folder}. Run: python scripts/download_data.py"
            )
        paths = []
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(p) as img:
                    img.verify()
                paths.append(p)
            except Exception:
                continue
        result[class_name] = paths
    return result


def resize_image(src: Path, dest: Path, size: int = config.IMAGE_SIZE) -> None:
    """Load, convert to RGB, resize to square, and save as JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        img.save(dest, format="JPEG", quality=90)


def stratified_split(
    class_images: dict[str, list[Path]],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
    max_per_class: int | None = None,
) -> dict[str, list[dict]]:
    """
    Stratified 80/10/10 split.

    Returns dict with keys train/val/test; each value is a list of
    {"path": relative_posix, "label": int, "class_name": str}.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for class_name, paths in class_images.items():
        items = list(paths)
        rng.shuffle(items)
        if max_per_class is not None:
            items = items[:max_per_class]
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder goes to test so counts add up
        buckets = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }
        label = config.CLASS_TO_IDX[class_name]
        for split_name, split_paths in buckets.items():
            for p in split_paths:
                splits[split_name].append(
                    {
                        "path": p.as_posix(),
                        "label": label,
                        "class_name": class_name,
                    }
                )

    for split_name in splits:
        rng.shuffle(splits[split_name])
    return splits


def materialize_processed_splits(
    splits: dict[str, list[dict]],
    processed_dir: Path | None = None,
    image_size: int = config.IMAGE_SIZE,
) -> dict[str, list[dict]]:
    """
    Copy/resize images into processed/{split}/{class}/ and rewrite paths
    to be relative to PROJECT_ROOT.
    """
    processed_dir = processed_dir or config.PROCESSED_DIR
    if processed_dir.exists():
        # Keep splits.json if present; clear image folders
        for split_name in ("train", "val", "test"):
            split_dir = processed_dir / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    rewritten: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split_name, records in splits.items():
        for rec in records:
            src = Path(rec["path"])
            class_name = rec["class_name"]
            dest = processed_dir / split_name / class_name / src.name
            resize_image(src, dest, size=image_size)
            rel = dest.relative_to(config.PROJECT_ROOT).as_posix()
            rewritten[split_name].append(
                {
                    "path": rel,
                    "label": rec["label"],
                    "class_name": class_name,
                }
            )
    return rewritten


def save_splits(splits: dict[str, list[dict]], path: Path | None = None) -> Path:
    path = path or config.SPLITS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: len(v) for k, v in splits.items()}
    payload = {"summary": summary, "splits": splits}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_splits(path: Path | None = None) -> dict[str, list[dict]]:
    path = path or config.SPLITS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Splits not found at {path}. Run: python scripts/prepare_data.py"
        )
    payload = json.loads(path.read_text())
    return payload["splits"]


def prepare_dataset(
    max_per_class: int | None = None,
    image_size: int = config.IMAGE_SIZE,
) -> dict[str, int]:
    """End-to-end preprocess: list → split → resize → save manifest."""
    class_images = list_class_images()
    splits = stratified_split(class_images, max_per_class=max_per_class)
    processed = materialize_processed_splits(splits, image_size=image_size)
    save_splits(processed)
    return {k: len(v) for k, v in processed.items()}
