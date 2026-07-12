"""Shared core prediction logic, used by api/main.py.

Kept separate from the FastAPI route handlers so the "given an image path,
return the model's decision" behavior -- quality gate, top-k prediction,
domain-similarity OOD check -- has one place to live if another surface
(the Streamlit admin tool, a future backend) ever needs the same decision
logic without duplicating what counts as "uncertain" vs "unknown".
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

from .config import (
    CONFIDENCE_FLOOR,
    IMG_SIZE,
    MAX_WHITE_BG,
    MIN_WHITE_BG,
    MODEL_DIR,
    OOD_FILE,
    REJECT_MAX_WHITE_BG,
    REJECT_MIN_WHITE_BG,
)
from .data import load_labels

_model = None
_labels: List[str] = []
_embedder = None
_ood_feats = None
_ood_threshold = None


def load_once() -> None:
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


def status() -> Dict[str, object]:
    """For health checks -- reports state without forcing a load."""
    return {
        "model_loaded": _model is not None,
        "num_classes": len(_labels),
        "ood_enabled": _ood_feats is not None,
    }


def leaf_scan_quality(path: str) -> str:
    """Classify upload quality by near-white-background fraction into three
    bands: "ok" (matches the training domain), "warn" (leaf-like but
    off-format -- still worth a prediction, flagged as less reliable), or
    "reject" (doesn't look like an attempted leaf photo at all -- caller
    should skip the model entirely)."""
    img = cv2.imread(path)
    if img is None:
        return "reject"
    white = float((img > 200).all(2).mean())
    if MIN_WHITE_BG <= white <= MAX_WHITE_BG:
        return "ok"
    if REJECT_MIN_WHITE_BG <= white <= REJECT_MAX_WHITE_BG:
        return "warn"
    return "reject"


def predict_topk(path: str, k: int = 3) -> Dict[str, object]:
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


def domain_similarity(arr: np.ndarray) -> Optional[float]:
    if _embedder is None or _ood_feats is None:
        return None
    emb = _embedder.predict(arr, verbose=0)[0]
    n = np.linalg.norm(emb)
    emb = emb / (n if n else 1.0)
    return float((_ood_feats @ emb).max())


def predict_image(path: str) -> Dict[str, object]:
    """End-to-end: quality gate -> prediction -> domain-similarity decision.

    Returns {"quality": "reject"} with no prediction fields if the image
    doesn't look like an attempted leaf photo at all -- the caller decides
    what that means for its own response (api/main.py raises 422), since
    that's presentation policy, not core inference.
    """
    load_once()
    quality = leaf_scan_quality(path)
    if quality == "reject":
        return {"quality": "reject"}

    pred = predict_topk(path, k=3)
    sim = domain_similarity(pred["array"])

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

    return {
        "species": pred["species"],
        "confidence": pred["confidence"],
        "top_k": pred["top_k"],
        "decision": decision,
        "quality": quality,
        "domain_similarity": sim,
        "num_classes": len(_labels),
    }
