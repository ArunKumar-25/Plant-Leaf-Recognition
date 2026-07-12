"""Gradio app entrypoint for Hugging Face Spaces.

Backend for web/identify.html when the primary FastAPI host is unavailable
-- wraps the same plantify.inference.predict_image() logic api/main.py
uses, so results match exactly regardless of which backend actually
served a given request.

This file plus src/plantify/{inference,config,data}.py and artifacts/
(model/, class_labels.json, ood.npz) are the full set of files a Space
deployment needs -- no training-time dependencies (data/, scripts/).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import gradio as gr

from plantify import inference


def identify(image_path: str | None) -> dict:
    if image_path is None:
        return {"quality": "reject"}
    return inference.predict_image(image_path)


demo = gr.Interface(
    fn=identify,
    inputs=gr.Image(type="filepath", label="Leaf photo"),
    outputs=gr.JSON(label="Result"),
    title="Plantify Leaf Identifier",
    description="Upload a single leaf on a plain, well-lit background.",
    api_name="identify",
)

if __name__ == "__main__":
    demo.launch(show_error=True)
