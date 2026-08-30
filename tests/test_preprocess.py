"""Unit tests for data preprocessing."""

from src import config
from src.data.preprocess import (
    list_class_images,
    resize_image,
    stratified_split,
)


def test_list_class_images(synthetic_raw):
    images = list_class_images(synthetic_raw)
    assert set(images.keys()) == {"cat", "dog"}
    assert len(images["cat"]) == 20
    assert len(images["dog"]) == 20


def test_stratified_split_ratios(synthetic_raw):
    images = list_class_images(synthetic_raw)
    splits = stratified_split(images, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == 40
    # Roughly 80/10/10
    assert len(splits["train"]) >= 28
    assert len(splits["val"]) >= 2
    assert len(splits["test"]) >= 2
    labels_train = {r["label"] for r in splits["train"]}
    assert labels_train == {0, 1}


def test_resize_image(synthetic_raw, tmp_path):
    images = list_class_images(synthetic_raw)
    src = images["cat"][0]
    dest = tmp_path / "out.jpg"
    resize_image(src, dest, size=config.IMAGE_SIZE)
    assert dest.exists()
    from PIL import Image

    with Image.open(dest) as img:
        assert img.size == (config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert img.mode == "RGB"
