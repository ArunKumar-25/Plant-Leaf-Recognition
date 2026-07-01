"""Security-focused tests for API upload validation paths."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


def _jpeg_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG")
    return buf.getvalue()


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


# --- _leaf_scan_quality: the three-band upload quality gate ---


def test_leaf_scan_quality_ok_band(tmp_path):
    # Mostly white background (a real scan's dominant color), within the
    # "ok" band (MIN_WHITE_BG..MAX_WHITE_BG).
    array = np.full((64, 64, 3), 255, dtype=np.uint8)
    array[20:44, 20:44] = 40  # a dark "leaf" occupying part of the frame
    path = tmp_path / "ok.jpg"
    path.write_bytes(_jpeg_bytes(array))

    assert api_main._leaf_scan_quality(str(path)) == "ok"


def test_leaf_scan_quality_reject_band_no_plain_background(tmp_path):
    # Solid black -- essentially zero near-white pixels, far below even the
    # "warn" band's lower bound. Not an attempted leaf scan at all.
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    path = tmp_path / "reject.jpg"
    path.write_bytes(_jpeg_bytes(array))

    assert api_main._leaf_scan_quality(str(path)) == "reject"


def test_leaf_scan_quality_reject_band_undecodable_file(tmp_path):
    path = tmp_path / "not-an-image.jpg"
    path.write_bytes(b"this is not image data")
    assert api_main._leaf_scan_quality(str(path)) == "reject"


def test_leaf_scan_quality_warn_band_borderline(tmp_path):
    # ~25% white -- inside the reject..ok gap (REJECT_MIN_WHITE_BG=0.15 up
    # to MIN_WHITE_BG=0.40): a busier background than the training domain,
    # but not so cluttered it looks like an unrelated photo.
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:16, :] = 255
    path = tmp_path / "warn.jpg"
    path.write_bytes(_jpeg_bytes(array))

    assert api_main._leaf_scan_quality(str(path)) == "warn"


def test_predict_rejects_non_leaf_photo_without_running_model(monkeypatch):
    client = _client_without_model_load(monkeypatch)
    monkeypatch.setattr(api_main, "_leaf_scan_quality", lambda _path: "reject")

    called = {"predict": False}

    def _fail_if_called(*_a, **_k):
        called["predict"] = True
        raise AssertionError("model should never run for a rejected upload")

    monkeypatch.setattr(api_main, "_predict_topk", _fail_if_called)

    array = np.full((64, 64, 3), 255, dtype=np.uint8)
    response = client.post(
        "/predict",
        files={"file": ("photo.jpg", _jpeg_bytes(array), "image/jpeg")},
    )

    assert response.status_code == 422
    assert "doesn't look like a leaf" in response.json()["detail"]
    assert called["predict"] is False
