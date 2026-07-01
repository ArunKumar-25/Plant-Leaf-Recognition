"""Security-focused tests for API upload validation paths."""

from fastapi.testclient import TestClient

import api.main as api_main


def _client_without_model_load(monkeypatch):
    monkeypatch.setattr(api_main, "_load_once", lambda: None)
    return TestClient(api_main.app)


def test_rate_limit_rejects_burst(monkeypatch):
    monkeypatch.setattr(api_main, "RATE_LIMIT_REQUESTS_PER_MINUTE", 1)
    api_main._request_windows.clear()
    client = _client_without_model_load(monkeypatch)

    first = client.post(
        "/predict",
        files={"file": ("leaf.txt", b"not-an-image", "text/plain")},
    )
    second = client.post(
        "/predict",
        files={"file": ("leaf.txt", b"not-an-image", "text/plain")},
    )

    assert first.status_code == 415
    assert second.status_code == 429
    assert second.json()["detail"] == "Too many requests"


def test_plantnet_fallback_is_opt_in(monkeypatch):
    monkeypatch.setattr(api_main, "PLANTNET_PUBLIC_FALLBACK_ENABLED", False)
    monkeypatch.setattr(api_main.plantnet_client, "identify", lambda _: (_ for _ in ()).throw(AssertionError))

    assert api_main._maybe_consult_plantnet("leaf.jpg", "unknown") is None


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
