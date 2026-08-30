"""API endpoint tests."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    import src.inference as inference

    inference._MODEL = None
    inference._MODEL_PATH = None

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_predict_success(trained_model, sample_image_bytes):
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("cat.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"cat", "dog"}
    assert "probabilities" in body


def test_predict_rejects_non_image(trained_model):
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_metrics_endpoint(trained_model):
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
