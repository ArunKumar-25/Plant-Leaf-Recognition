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

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from plantify import inference, plantnet_client
from plantify.config import (
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    GITHUB_BRANCH,
    GITHUB_CONTRIB_TOKEN,
    GITHUB_REPO,
    MAX_UPLOAD_BYTES,
    PLANTNET_DAILY_CAP,
    PLANTNET_PUBLIC_FALLBACK_ENABLED,
    PLANTNET_STAGE_THRESHOLD,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    STAGING_MAX_RETRIES,
)
from plantify.data import PENDING_MANIFEST, build_pending_row

logging.getLogger("tensorflow").setLevel(logging.ERROR)
_staging_logger = logging.getLogger("plantify.staging")
_api_logger = logging.getLogger("plantify.api")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    inference.load_once()
    yield


app = FastAPI(title="Plantify Leaf API", version="2.0.0", lifespan=lifespan)

allow_credentials = "*" not in CORS_ALLOW_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort catch-all: log the full traceback server-side, never leak
    internal detail (stack trace, exception message) to the client."""
    _api_logger.exception(
        "Unhandled exception on %s %s: %r", request.method, request.url.path, exc
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


_plantnet_call_date: datetime.date | None = None
_plantnet_call_count = 0
_request_windows: Dict[str, List[float]] = {}


def _client_id(request: Request) -> str:
    # Take the LAST hop, not the first: reverse proxies (including Azure
    # Container Apps' ingress) append the real peer IP to this header rather
    # than replacing it, so the first entry is whatever the client itself
    # claimed. Trusting it would let a client defeat per-IP rate limiting
    # just by sending a different fake X-Forwarded-For on every request.
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.rsplit(",", 1)[-1].strip()
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


def _maybe_consult_plantnet(temp_path: str, decision: str) -> Dict[str, object] | None:
    """Ask Pl@ntNet for a second opinion on `unknown` or `uncertain` results,
    up to a daily cap (shared quota with the Streamlit admin tool's manual
    use). `uncertain` is included deliberately -- the model's raw softmax
    confidence can be severely miscalibrated on out-of-distribution photos
    (confirmed live: a real upload scored 100% confidence on the wrong
    species while the OOD guard correctly flagged it "uncertain"), so
    without this, exactly the cases most worth a second opinion never got
    one. Never raises — a Pl@ntNet outage degrades to "no second opinion",
    not a 500.
    """
    if decision not in ("unknown", "uncertain") or not PLANTNET_PUBLIC_FALLBACK_ENABLED:
        return None

    global _plantnet_call_date, _plantnet_call_count
    today = datetime.date.today()
    if _plantnet_call_date != today:
        _plantnet_call_date = today
        _plantnet_call_count = 0
    if _plantnet_call_count >= PLANTNET_DAILY_CAP:
        return None
    _plantnet_call_count += 1

    try:
        result = plantnet_client.identify(temp_path)
    except Exception:
        return None
    if result.get("error") or not result.get("results"):
        return None
    return result["results"][0]


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
def health() -> JSONResponse:
    try:
        inference.load_once()
    except Exception:
        _api_logger.exception("Model failed to load")
        return JSONResponse(
            status_code=503, content={"status": "error", "detail": "model failed to load"}
        )
    return JSONResponse({"status": "ok", **inference.status()})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    inference.load_once()
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
        try:
            result = inference.predict_image(temp_path)
        except Exception:
            _api_logger.exception("Model inference failed")
            raise HTTPException(status_code=503, detail="Model temporarily unavailable") from None

        quality = result["quality"]
        if quality == "reject":
            raise HTTPException(
                status_code=422,
                detail="This doesn't look like a leaf photo. Try a clear photo of a single leaf on a plain background.",
            )

        decision = result["decision"]
        sim = result["domain_similarity"]

        payload = {
            "species": result["species"],
            "confidence": result["confidence"],
            "top_k": result["top_k"],
            "decision": decision,
            "quality": quality,
            "domain_similarity": sim,
            "num_classes": result["num_classes"],
        }
        if quality == "warn":
            payload["quality_warning"] = (
                "This photo doesn't closely match the clean, single-leaf scans the model was trained "
                "on, so this result may be less reliable than usual."
            )

        plantnet_top = _maybe_consult_plantnet(temp_path, decision)
        if plantnet_top is not None:
            plantnet_top["staged"] = plantnet_top["score"] >= PLANTNET_STAGE_THRESHOLD
            payload["plantnet"] = plantnet_top
            if plantnet_top["score"] >= PLANTNET_STAGE_THRESHOLD:
                row = build_pending_row(
                    predicted_species=result["species"],
                    model_confidence=result["confidence"],
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
