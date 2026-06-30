"""Security-focused tests for API upload validation paths."""

from fastapi.testclient import TestClient

import api.main as api_main


def _client_without_model_load(monkeypatch):
    monkeypatch.setattr(api_main, "_load_once", lambda: None)
    return TestClient(api_main.app)


def test_rejects_unsupported_content_type(monkeypatch):
    client = _client_without_model_load(monkeypatch)

    response = client.post(
        "/predict",
        files={"file": ("leaf.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported content type"


def test_rejects_unsupported_file_extension(monkeypatch):
    client = _client_without_model_load(monkeypatch)

    response = client.post(
        "/predict",
        files={"file": ("leaf.gif", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file extension"


def test_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
    client = _client_without_model_load(monkeypatch)

    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", b"01234567890", "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"
