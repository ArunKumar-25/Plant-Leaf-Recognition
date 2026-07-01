"""FastAPI inference service for Plantify website and backend integrations."""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

from src.plantify import plantnet_client
from src.plantify.config import CONFIDENCE_FLOOR, IMG_SIZE, MODEL_DIR, OOD_FILE
from src.plantify.data import PENDING_MANIFEST, build_pending_row, load_labels

logging.getLogger("tensorflow").setLevel(logging.ERROR)
_staging_logger = logging.getLogger("plantify.staging")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

PLANTNET_DAILY_CAP = int(os.getenv("PLANTNET_DAILY_CAP", "300"))
PLANTNET_PUBLIC_FALLBACK_ENABLED = os.getenv("PLANTNET_PUBLIC_FALLBACK_ENABLED", "").lower() in {
    "1",
    "true",
    "yes",
}
PLANTNET_STAGE_THRESHOLD = float(os.getenv("PLANTNET_STAGE_THRESHOLD", "0.70"))
GITHUB_CONTRIB_TOKEN = os.getenv("GITHUB_CONTRIB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "contributions")
STAGING_MAX_RETRIES = int(os.getenv("STAGING_MAX_RETRIES", "3"))
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "30"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_once()
    yield


app = FastAPI(title="Plantify Leaf API", version="2.0.0", lifespan=lifespan)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
allow_credentials = "*" not in allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_labels: List[str] = []
_embedder = None
_ood_feats = None
_ood_threshold = None
_plantnet_call_date: datetime.date | None = None
_plantnet_call_count = 0
_request_windows: Dict[str, List[float]] = {}


def _load_once() -> None:
    global _model, _labels, _embedder, _ood_feats, _ood_threshold
    if _model is not None:
        return

    _model = keras.models.load_model(MODEL_DIR)
    _labels = load_labels()

    try:
        base = next(
            layer for layer in _model.layers if "mobilenet" in layer.name.lower()
        )
        rescale = next(
            layer for layer in _model.layers if "rescal" in layer.name.lower()
        )
        inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        _embedder = keras.Model(inp, base(rescale(inp)))
    except Exception:
        _embedder = None

    if os.path.exists(OOD_FILE):
        d = np.load(OOD_FILE)
        _ood_feats = d["feats"]
        _ood_threshold = float(d["threshold"])


def _looks_like_leaf_scan(path: str) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    white = float((img > 200).all(2).mean())
    return 0.40 <= white <= 0.97


def _predict_topk(path: str, k: int = 3) -> Dict[str, object]:
    img = keras_image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = (
        keras_image.img_to_array(img)
        .reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        .astype("float32")
    )
    out = _model.predict(arr, verbose=0)[0]

    order = np.argsort(out)[::-1][:k]
    top_k = []
    for idx in order:
        label = _labels[int(idx)] if int(idx) < len(_labels) else "unknown"
        top_k.append({"species": label, "confidence": float(out[int(idx)])})

    best_idx = int(order[0]) if len(order) else 0
    best_conf = float(out[best_idx]) if len(order) else 0.0
    species = _labels[best_idx] if best_idx < len(_labels) else "unknown"
    return {"species": species, "confidence": best_conf, "top_k": top_k, "array": arr}


def _domain_similarity(arr: np.ndarray) -> float | None:
    if _embedder is None or _ood_feats is None:
        return None
    emb = _embedder.predict(arr, verbose=0)[0]
    n = np.linalg.norm(emb)
    emb = emb / (n if n else 1.0)
    return float((_ood_feats @ emb).max())


def _client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_id: str) -> None:
    if RATE_LIMIT_REQUESTS_PER_MINUTE <= 0:
        return

    now = time.monotonic()
    cutoff = now - 60
    recent = [stamp for stamp in _request_windows.get(client_id, []) if stamp > cutoff]
    if len(recent) >= RATE_LIMIT_REQUESTS_PER_MINUTE:
        _request_windows[client_id] = recent
        raise HTTPException(status_code=429, detail="Too many requests")
    recent.append(now)
    _request_windows[client_id] = recent


def _maybe_consult_plantnet(temp_path: str, decision: str) -> tuple[Dict[str, object] | None, Dict[str, object]]:
    """Ask Pl@ntNet for a second opinion, but only on `unknown` and only up
    to a daily cap (shared quota with the Streamlit admin tool's manual use).
    Never raises — a Pl@ntNet outage degrades to "no second opinion", not a 500.

    Returns (top_result_or_None, debug_info) — debug_info is temporary, for
    diagnosing why the feature isn't showing up in production.
    """
    debug: Dict[str, object] = {
        "decision": decision,
        "enabled": PLANTNET_PUBLIC_FALLBACK_ENABLED,
        "has_key": bool(plantnet_client.get_api_key()),
    }
    if decision != "unknown" or not PLANTNET_PUBLIC_FALLBACK_ENABLED:
        return None, debug

    global _plantnet_call_date, _plantnet_call_count
    today = datetime.date.today()
    if _plantnet_call_date != today:
        _plantnet_call_date = today
        _plantnet_call_count = 0
    if _plantnet_call_count >= PLANTNET_DAILY_CAP:
        debug["daily_cap_hit"] = "%d/%d" % (_plantnet_call_count, PLANTNET_DAILY_CAP)
        return None, debug
    _plantnet_call_count += 1

    try:
        result = plantnet_client.identify(temp_path)
    except Exception as exc:
        debug["exception"] = repr(exc)
        return None, debug
    debug["raw_result"] = result
    if result.get("error") or not result.get("results"):
        return None, debug
    return result["results"][0], debug


def _stage_candidate(image_bytes: bytes, row: Dict[str, object]) -> None:
    """Best-effort: push a candidate image + manifest row to the dedicated
    `contributions` branch via the GitHub Contents API. No-ops if staging
    isn't configured (GITHUB_CONTRIB_TOKEN/GITHUB_REPO unset) — the live API
    has no durable local disk, so GitHub itself is the only durable store.
    Never raises into the request path; staging failures are logged only.
    """
    if not (GITHUB_CONTRIB_TOKEN and GITHUB_REPO):
        return

    headers = {
        "Authorization": f"Bearer {GITHUB_CONTRIB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    base_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents"

    def _content_get(path: str):
        return requests.get(
            f"{base_url}/{path}",
            headers=headers,
            params={"ref": GITHUB_BRANCH},
            timeout=10,
        )

    def _content_put(path: str, body: Dict[str, object]):
        return requests.put(
            f"{base_url}/{path}", headers=headers, json=body, timeout=10
        )

    def _delete_staged_image(path: str) -> None:
        try:
            get_resp = _content_get(path)
            if get_resp.status_code != 200:
                return
            sha = get_resp.json().get("sha")
            if not sha:
                return
            resp = requests.delete(
                f"{base_url}/{path}",
                headers=headers,
                json={
                    "message": f"chore(contrib): rollback orphan candidate image {row['id']}",
                    "sha": sha,
                    "branch": GITHUB_BRANCH,
                },
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                _staging_logger.warning(
                    "Rollback image delete failed for %s: %s", path, resp.status_code
                )
        except Exception:
            _staging_logger.exception("Rollback image delete errored for %s", path)

    try:
        image_path = row["image"]
        image_resp = _content_put(
            image_path,
            {
                "message": f"chore(contrib): stage candidate image {row['id']}",
                "content": base64.b64encode(image_bytes).decode("ascii"),
                "branch": GITHUB_BRANCH,
            },
        )
        if image_resp.status_code not in (200, 201):
            _staging_logger.warning(
                "Failed to stage image %s: %s", row.get("id"), image_resp.status_code
            )
            return

        staged = False
        for attempt in range(STAGING_MAX_RETRIES):
            get_resp = _content_get(PENDING_MANIFEST)
            if get_resp.status_code == 200:
                current = get_resp.json()
                existing = base64.b64decode(current["content"]).decode("utf-8")
                sha = current.get("sha")
            elif get_resp.status_code == 404:
                existing = ""
                sha = None
            else:
                _staging_logger.warning(
                    "Failed reading manifest before append for %s: %s",
                    row.get("id"),
                    get_resp.status_code,
                )
                break

            body = {
                "message": f"chore(contrib): record candidate {row['id']}",
                "content": base64.b64encode(
                    (existing + json.dumps(row) + "\n").encode("utf-8")
                ).decode("ascii"),
                "branch": GITHUB_BRANCH,
            }
            if sha:
                body["sha"] = sha
            put_resp = _content_put(PENDING_MANIFEST, body)
            if put_resp.status_code in (200, 201):
                staged = True
                break
            if put_resp.status_code == 409:
                time.sleep(0.25 * (attempt + 1))
                continue
            _staging_logger.warning(
                "Failed writing manifest for %s: %s",
                row.get("id"),
                put_resp.status_code,
            )
            break

        if not staged:
            _delete_staged_image(image_path)
            _staging_logger.warning(
                "Candidate %s was not staged into manifest", row.get("id")
            )
    except Exception:
        _staging_logger.exception("Failed to stage candidate %s", row.get("id"))


@app.get("/health")
def health() -> Dict[str, object]:
    _load_once()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_labels),
        "ood_enabled": _ood_feats is not None,
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    _load_once()
    _check_rate_limit(_client_id(request))

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported content type")

    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file extension")

    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(body)
        temp_path = temp.name

    try:
        Image.open(temp_path).verify()
    except Exception as exc:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    try:
        quality_ok = _looks_like_leaf_scan(temp_path)
        pred = _predict_topk(temp_path, k=3)
        sim = _domain_similarity(pred["array"])

        if sim is None:
            decision = "ok" if pred["confidence"] >= CONFIDENCE_FLOOR else "unknown"
        else:
            high = _ood_threshold
            low = max(0.60, high - 0.15)
            if sim >= high:
                decision = "ok"
            elif sim >= low:
                decision = "uncertain"
            else:
                decision = "unknown"

        payload = {
            "species": pred["species"],
            "confidence": pred["confidence"],
            "top_k": pred["top_k"],
            "decision": decision,
            "quality_ok": quality_ok,
            "domain_similarity": sim,
            "num_classes": len(_labels),
        }

        plantnet_top, plantnet_debug = _maybe_consult_plantnet(temp_path, decision)
        payload["_debug_plantnet"] = plantnet_debug
        if plantnet_top is not None:
            payload["plantnet"] = plantnet_top
            if plantnet_top["score"] >= PLANTNET_STAGE_THRESHOLD:
                row = build_pending_row(
                    predicted_species=pred["species"],
                    model_confidence=pred["confidence"],
                    domain_similarity=sim,
                    decision=decision,
                    plantnet_species=plantnet_top["name"],
                    plantnet_confidence=plantnet_top["score"],
                    plantnet_common=plantnet_top.get("common", ""),
                    source="api",
                    ext=suffix,
                )
                _stage_candidate(body, row)

        return JSONResponse(payload)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
