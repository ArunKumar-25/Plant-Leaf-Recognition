"""Evaluate the currently *committed* model (artifacts/model/) against data/,
without retraining. Used by the weekly self-retrain workflow to get a "before"
baseline and an "after" score for the regression gate (see regression_gate.py).

Uses the same deterministic split (seed=42, same per-class percentages) as
src/plantify/training.py's train_and_evaluate(), so a run before promoting new
candidates and a run after are methodologically consistent — though the actual
test set does grow when new images land in data/ between the two runs. That's
an expected, normal trait of evaluating a continuously-growing dataset, not a
bug; regression_gate.py's tolerance absorbs the resulting run-to-run noise.

Usage:
    python scripts/evaluate_model.py --out artifacts/reports/_baseline_metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
from tensorflow import keras

from plantify import data
from plantify.config import IMG_SIZE, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def _load_test_split() -> tuple[np.ndarray, np.ndarray, list[str]]:
    folders = [f for f in data.list_class_folders() if len(data.images_in(f)) >= data.MIN_PER_CLASS]
    classes = [data.species_of(f) for f in folders]

    rng = np.random.default_rng(SEED)
    x_test, y_test = [], []
    for idx, folder in enumerate(folders):
        images = []
        for name in data.images_in(folder):
            raw = cv2.imread(os.path.join(data.DATA_DIR, folder, name), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
            images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        rng.shuffle(images)
        n_test = max(1, round(len(images) * 0.20))
        for img in images[:n_test]:
            x_test.append(img)
            y_test.append(idx)

    if not x_test:
        return np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), np.zeros((0,), dtype="int32"), classes
    return np.array(x_test, dtype="float32"), np.array(y_test, dtype="int32"), classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to write metrics JSON")
    args = parser.parse_args()

    x_test, y_test, classes = _load_test_split()
    if len(y_test) == 0:
        logger.info("No test images found under data/ — nothing to evaluate.")
        json.dump({"accuracy": None, "per_class": {}, "n_test": 0}, open(args.out, "w", encoding="utf-8"))
        return 0

    model = keras.models.load_model(MODEL_DIR)
    pred = model.predict(x_test, verbose=0).argmax(axis=1)

    accuracy = float((pred == y_test).mean())
    per_class = {}
    per_class_support = {}
    for idx, species in enumerate(classes):
        mask = y_test == idx
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((pred[mask] == idx).sum())
        per_class[species] = correct / total
        # Raw counts alongside the ratio -- regression_gate.py uses these to
        # test whether a recall drop is statistically significant rather than
        # just comparing ratios, since ~15 test images/class means a single
        # flipped prediction swings the ratio by ~6.7% on its own.
        per_class_support[species] = {"correct": correct, "total": total}

    metrics = {
        "accuracy": accuracy,
        "per_class": per_class,
        "per_class_support": per_class_support,
        "n_test": int(len(y_test)),
        "classes": classes,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Evaluated on %d test images: accuracy=%.4f", len(y_test), accuracy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
