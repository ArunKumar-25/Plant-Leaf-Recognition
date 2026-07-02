"""Security-focused tests for API upload validation paths."""

import datetime
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


def _mock_unknown_prediction(monkeypatch):
    monkeypatch.setattr(api_main, "_leaf_scan_quality", lambda _path: "ok")
    monkeypatch.setattr(
        api_main,
        "_predict_topk",
        lambda *_a, **_k: {"species": "Acer", "confidence": 0.1, "top_k": [], "array": np.zeros((1, 1, 1, 1))},
    )
    monkeypatch.setattr(api_main, "_domain_similarity", lambda *_a, **_k: None)


def _mock_uncertain_prediction(monkeypatch):
    # A raw-confidence "ok" prediction whose OOD domain similarity lands in
    # the "uncertain" band (low <= sim < high) -- the case a model can be
    # 100% confident and still wrong on (a real Sorbus intermedia leaf
    # misclassified as Quercus), which is exactly why this band also
    # deserves a Pl@ntNet second opinion.
    monkeypatch.setattr(api_main, "_leaf_scan_quality", lambda _path: "ok")
    monkeypatch.setattr(
        api_main,
        "_predict_topk",
        lambda *_a, **_k: {"species": "Quercus", "confidence": 1.0, "top_k": [], "array": np.zeros((1, 1, 1, 1))},
    )
    monkeypatch.setattr(api_main, "_ood_threshold", 0.90)
    monkeypatch.setattr(api_main, "_domain_similarity", lambda *_a, **_k: 0.80)


def test_plantnet_result_below_threshold_marked_not_staged(monkeypatch):
    client = _client_without_model_load(monkeypatch)
    _mock_unknown_prediction(monkeypatch)
    monkeypatch.setattr(api_main, "PLANTNET_STAGE_THRESHOLD", 0.70)
    monkeypatch.setattr(
        api_main,
        "_maybe_consult_plantnet",
        lambda *_a, **_k: {"name": "Tilia platyphyllos", "common": "", "score": 0.41},
    )

    array = np.full((64, 64, 3), 255, dtype=np.uint8)
    response = client.post("/predict", files={"file": ("leaf.jpg", _jpeg_bytes(array), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["plantnet"]["staged"] is False


def test_plantnet_result_above_threshold_marked_staged(monkeypatch):
    client = _client_without_model_load(monkeypatch)
    _mock_unknown_prediction(monkeypatch)
    monkeypatch.setattr(api_main, "PLANTNET_STAGE_THRESHOLD", 0.70)
    monkeypatch.setattr(api_main, "_stage_candidate", lambda *_a, **_k: None)
    monkeypatch.setattr(
        api_main,
        "_maybe_consult_plantnet",
        lambda *_a, **_k: {"name": "Tilia platyphyllos", "common": "", "score": 0.85},
    )

    array = np.full((64, 64, 3), 255, dtype=np.uint8)
    response = client.post("/predict", files={"file": ("leaf.jpg", _jpeg_bytes(array), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["plantnet"]["staged"] is True


def test_plantnet_is_consulted_on_uncertain_decision(monkeypatch):
    client = _client_without_model_load(monkeypatch)
    _mock_uncertain_prediction(monkeypatch)
    monkeypatch.setattr(api_main, "PLANTNET_STAGE_THRESHOLD", 0.70)
    monkeypatch.setattr(api_main, "_stage_candidate", lambda *_a, **_k: None)
    monkeypatch.setattr(
        api_main,
        "_maybe_consult_plantnet",
        lambda *_a, **_k: {"name": "Sorbus intermedia", "common": "", "score": 0.85},
    )

    array = np.full((64, 64, 3), 255, dtype=np.uint8)
    response = client.post("/predict", files={"file": ("leaf.jpg", _jpeg_bytes(array), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "uncertain"
    assert body["plantnet"]["staged"] is True


def test_maybe_consult_plantnet_fires_for_uncertain_decision(monkeypatch):
    monkeypatch.setattr(api_main, "PLANTNET_PUBLIC_FALLBACK_ENABLED", True)
    monkeypatch.setattr(api_main, "_plantnet_call_date", datetime.date.today())
    monkeypatch.setattr(api_main, "_plantnet_call_count", 0)
    monkeypatch.setattr(api_main, "PLANTNET_DAILY_CAP", 100)
    monkeypatch.setattr(
        api_main.plantnet_client,
        "identify",
        lambda _path: {"results": [{"name": "Sorbus intermedia", "common": "", "score": 0.85}]},
    )

    result = api_main._maybe_consult_plantnet("leaf.jpg", "uncertain")

    assert result is not None
    assert result["name"] == "Sorbus intermedia"


def test_maybe_consult_plantnet_still_skips_ok_decision(monkeypatch):
    monkeypatch.setattr(api_main, "PLANTNET_PUBLIC_FALLBACK_ENABLED", True)
    monkeypatch.setattr(
        api_main.plantnet_client, "identify", lambda _: (_ for _ in ()).throw(AssertionError)
    )

    assert api_main._maybe_consult_plantnet("leaf.jpg", "ok") is None
