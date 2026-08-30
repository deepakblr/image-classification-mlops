"""Unit tests for inference utilities."""

from PIL import Image

from src.inference import predict_bytes, predict_image, preprocess_image


def test_preprocess_image_shape(trained_model):
    img = Image.new("RGB", (300, 200), color=(10, 20, 30))
    tensor = preprocess_image(img)
    assert tuple(tensor.shape) == (1, 3, 224, 224)


def test_predict_image_keys(trained_model):
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    result = predict_image(img)
    assert result["label"] in {"cat", "dog"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"cat", "dog"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_predict_bytes(trained_model, sample_image_bytes):
    result = predict_bytes(sample_image_bytes)
    assert "prediction" in result
    assert result["label"] in {"cat", "dog"}
