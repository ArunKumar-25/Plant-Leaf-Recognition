"""Streamlit UI for Plantify inference, teaching, and retraining."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from . import data, plantnet_client
from .config import CONFIDENCE_FLOOR, IMG_SIZE, MODEL_DIR, OOD_FILE

logging.getLogger("tensorflow").setLevel(logging.ERROR)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}


def _model_version() -> float:
    pb = os.path.join(MODEL_DIR, "saved_model.pb")
    return os.path.getmtime(pb) if os.path.exists(pb) else 0.0


@st.cache_resource(show_spinner=False)
def load_cnn(version: float):
    _ = version
    return keras.models.load_model(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def load_embedder(version: float):
    _ = version
    model = load_cnn(version)
    try:
        base = next(layer for layer in model.layers if "mobilenet" in layer.name.lower())
        rescale = next(layer for layer in model.layers if "rescal" in layer.name.lower())
        inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        return keras.Model(inp, base(rescale(inp)))
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_ood(version: float):
    _ = version
    if not os.path.exists(OOD_FILE):
        return None
    values = np.load(OOD_FILE)
    return values["feats"], float(values["threshold"])


def domain_similarity(img_path: str, embedder, feats: np.ndarray) -> float:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    emb = embedder.predict(arr, verbose=0)[0]
    norm = np.linalg.norm(emb)
    emb = emb / (norm if norm else 1.0)
    return float((feats @ emb).max())


def prediction(img_path: str, model):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    out = model.predict(arr, verbose=0)
    idx = int(np.argmax(out, axis=-1)[0])
    return idx, float(out[0][idx])


def apply_mask(img_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (86, 255, 255))
    result = img.copy()
    result[mask == 0] = (255, 255, 255)
    masked_path = os.path.join(tempfile.gettempdir(), f"plantify_masked_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(masked_path, result)
    return masked_path


def looks_like_leaf_scan(img_path: str) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image")
    white = float((img > 200).all(2).mean())
    return 0.40 <= white <= 0.97


def validate_uploaded_image(uploaded) -> bytes | None:
    if getattr(uploaded, "type", None) not in ALLOWED_UPLOAD_TYPES:
        st.error("Unsupported image type. Please upload JPG, PNG, BMP, TIFF, or WEBP.")
        return None
    if getattr(uploaded, "size", 0) > MAX_UPLOAD_BYTES:
        st.error("Image is too large. Maximum allowed size is 8 MB.")
        return None

    image_bytes = uploaded.getvalue()
    try:
        Image.open(uploaded).verify()
    except Exception:
        st.error("Invalid image. Please upload a valid leaf photo.")
        return None
    finally:
        uploaded.seek(0)

    return image_bytes


def retrain() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/train_model.py"], capture_output=True, text=True)


def main() -> None:
    st.title("Plantify Leaf Classifier")
    st.caption("Upload a single leaf on plain background for best accuracy.")

    labels = data.load_labels()
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

    if st.button("Predict"):
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            image_bytes = validate_uploaded_image(uploaded)
            if image_bytes is None:
                return

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(Image.open(uploaded), caption="Uploaded", width=130)

            uploaded.seek(0)
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            with col2:
                try:
                    model = load_cnn(_model_version())
                    quality_ok = looks_like_leaf_scan(tmp_path)
                    used = tmp_path
                    if not quality_ok:
                        used = apply_mask(tmp_path)
                        with col1:
                            st.image(used, caption="Background removed", width=130)

                    idx, acc = prediction(used, model)
                except Exception:
                    st.error("Could not process this image. Please try another leaf photo.")
                    return
                if idx >= len(labels):
                    idx = 0

                sim = None
                embedder = load_embedder(_model_version())
                ood_values = load_ood(_model_version())
                if embedder is not None and ood_values is not None:
                    feats, thr_high = ood_values
                    thr_low = max(0.60, thr_high - 0.15)
                    sim = domain_similarity(used, embedder, feats)
                else:
                    thr_high = CONFIDENCE_FLOOR
                    thr_low = CONFIDENCE_FLOOR

                if not quality_ok:
                    st.warning("Image quality/background is outside ideal training conditions.")

                if sim is None:
                    decision = "ok" if acc >= CONFIDENCE_FLOOR else "unknown"
                elif sim >= thr_high:
                    decision = "ok"
                elif sim >= thr_low:
                    decision = "uncertain"
                else:
                    decision = "unknown"

                if decision == "ok":
                    st.success("Prediction: %s (%.2f%%)" % (labels[idx], acc * 100))
                elif decision == "uncertain":
                    st.warning("Uncertain match: %s (feature match %.0f%%)." % (labels[idx], sim * 100))
                else:
                    extra = " (closest match %.0f%%)" % (sim * 100) if sim is not None else ""
                    st.error("Leaf not recognized as known species%s." % extra)

                query = "+".join(labels[idx].split())
                st.write("[Learn more](https://www.google.com/search?q=%s+leaf)" % query)

                st.session_state["last_image"] = tmp_path
                st.session_state["last_pred"] = labels[idx]

    last = st.session_state.get("last_image")
    if last and os.path.exists(last):
        st.markdown("---")
        st.subheader("Teach the model")
        known = data.load_labels()
        default_idx = known.index(st.session_state["last_pred"]) if st.session_state.get("last_pred") in known else 0
        choice = st.selectbox("Correct species", known + ["Add new species..."], index=default_idx)

        species = choice
        if choice == "Add new species...":
            species = st.text_input("New species name").strip()

        if st.button("Add image to dataset"):
            if not species:
                st.warning("Enter species name first.")
            else:
                _, folder = data.save_contribution(last, species)
                count = len(data.images_in(folder))
                st.success("Saved to data/%s (now %d image(s))." % (folder, count))
                if count < data.MIN_PER_CLASS:
                    st.info("Add %d more image(s) before class can be trained." % (data.MIN_PER_CLASS - count))

        st.markdown("#### Second opinion (Pl@ntNet)")
        if not plantnet_client.get_api_key():
            st.caption("Set PLANTNET_API_KEY in environment or Streamlit secrets to enable.")
        elif st.button("Ask Pl@ntNet"):
            with st.spinner("Requesting Pl@ntNet..."):
                result = plantnet_client.identify(last)
            if result.get("error"):
                st.warning(plantnet_client.friendly_error(result["error"]))
            elif result.get("results"):
                for row in result["results"]:
                    common = (" (%s)" % row["common"]) if row["common"] else ""
                    st.write("- **%s**%s - %.0f%%" % (row["name"], common, row["score"] * 100))
            else:
                st.info("No result returned.")

    st.markdown("---")
    st.subheader("Retrain")
    st.caption("Retrain from full data/ including contributions.")
    if st.button("Retrain model now"):
        with st.spinner("Training..."):
            result = retrain()
        if result.returncode == 0:
            st.success("Retraining complete.")
            tail = "\n".join(result.stdout.strip().splitlines()[-8:])
            st.code(tail)
        else:
            st.error("Retraining failed.")
            st.code((result.stdout + "\n" + result.stderr).strip()[-2500:])


if __name__ == "__main__":
    main()
