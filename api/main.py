"""FastAPI inference service for Plantify website and backend integrations."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

from src.plantify.config import CONFIDENCE_FLOOR, IMG_SIZE, MODEL_DIR, OOD_FILE
from src.plantify.data import load_labels

logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = FastAPI(title="Plantify Leaf API", version="2.0.0")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
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


def _load_once() -> None:
    global _model, _labels, _embedder, _ood_feats, _ood_threshold
    if _model is not None:
        return

    _model = keras.models.load_model(MODEL_DIR)
    _labels = load_labels()

    try:
        base = next(layer for layer in _model.layers if "mobilenet" in layer.name.lower())
        rescale = next(layer for layer in _model.layers if "rescal" in layer.name.lower())
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
    arr = keras_image.img_to_array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
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


@app.on_event("startup")
def startup() -> None:
    _load_once()


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
async def predict(file: UploadFile = File(...)):
    _load_once()

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
        return JSONResponse(payload)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
