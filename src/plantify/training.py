"""Training pipeline for Plantify leaf classification model."""

from __future__ import annotations

import json
import os
import sys
from typing import List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from tensorflow.keras.optimizers import Adam

from . import data
from .config import IMG_SIZE, LABELS_FILE, MODEL_DIR, OOD_FILE, REPORTS_DIR

matplotlib.use("Agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Fine-tuning phase: fraction of backbone layers (from the end) to unfreeze,
# and the learning rate to unfreeze them at. Kept low/shallow since the
# training set is small (~750 images) — fine-tuning the whole backbone on
# that little data would overfit fast.
FINE_TUNE_LAYER_FRACTION = 0.2
FINE_TUNE_LR = 1e-5


def _load_folder(folder: str) -> List[np.ndarray]:
    output = []
    for name in data.images_in(folder):
        raw = cv2.imread(os.path.join(data.DATA_DIR, folder, name), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
        output.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return output


def _to_array(images: List[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    return np.array(images, dtype="float32")


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return features / norm


def _build_augmenter() -> Sequential:
    # Leaf scans have no canonical orientation, so flips/full rotation are
    # free signal, not distortion. Zoom/contrast add mild lighting/framing
    # robustness without risking cropping out the leaf.
    return Sequential(
        [
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.5),
            RandomZoom(0.1),
            RandomContrast(0.1),
        ],
        name="augmentation",
    )


def train_and_evaluate(use_augmentation: bool = True, use_fine_tune: bool = True) -> int:
    folders = [f for f in data.list_class_folders() if len(data.images_in(f)) >= data.MIN_PER_CLASS]
    if len(folders) < 2:
        print("Need >= 2 classes with >= %d images. Found: %s" % (data.MIN_PER_CLASS, folders))
        return 1

    classes = [data.species_of(f) for f in folders]
    print("Training classes (%d):" % len(classes), classes)

    rng = np.random.default_rng(SEED)
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    for idx, folder in enumerate(folders):
        images = _load_folder(folder)
        rng.shuffle(images)
        n = len(images)
        n_test = max(1, round(n * 0.20))
        n_val = max(1, round((n - n_test) * 0.15))

        test = images[:n_test]
        val = images[n_test : n_test + n_val]
        train = images[n_test + n_val :]
        if not train:
            train, val = (val or test), []

        for img in train:
            x_train.append(img)
            y_train.append(idx)
        for img in val:
            x_val.append(img)
            y_val.append(idx)
        for img in test:
            x_test.append(img)
            y_test.append(idx)

    x_train = _to_array(x_train)
    x_val = _to_array(x_val)
    x_test = _to_array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    print("train/val/test:", len(y_train), len(y_val), len(y_test))

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.5,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    head = Sequential(
        [
            Input((base.output_shape[-1],)),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(len(classes), activation="softmax"),
        ]
    )

    rescale = Rescaling(1.0 / 127.5, offset=-1.0)

    # Training graph: augmentation (when enabled) is only ever active while
    # a Sequential of preprocessing layers is called with training=True
    # implicitly during model.fit(); at inference (predict/evaluate) it's a
    # no-op regardless.
    train_inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    t = _build_augmenter()(train_inputs) if use_augmentation else train_inputs
    t = rescale(t)
    t = base(t, training=False)
    train_model = Model(train_inputs, head(t))

    # Inference graph shares the same base/head weights but skips
    # augmentation entirely, so nothing from the training pipeline can leak
    # into what gets saved and served.
    infer_inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    embedding_output = base(rescale(infer_inputs), training=False)
    infer_model = Model(infer_inputs, head(embedding_output))
    embed_model = Model(infer_inputs, embedding_output)

    val_data = (x_val, y_val) if len(y_val) else None

    print("Phase 1: training head with frozen backbone...")
    train_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callback1 = EarlyStopping(monitor="val_loss" if val_data else "loss", patience=10, restore_best_weights=True)
    history1 = train_model.fit(
        x_train,
        y_train,
        validation_data=val_data,
        epochs=60,
        batch_size=16,
        callbacks=[callback1],
        verbose=2,
    )

    history2 = None
    if use_fine_tune:
        print("Phase 2: fine-tuning top of the backbone...")
        n_layers = len(base.layers)
        cutoff = int(n_layers * (1 - FINE_TUNE_LAYER_FRACTION))
        base.trainable = True
        for layer in base.layers[:cutoff]:
            layer.trainable = False
        print("Unfroze %d/%d backbone layers." % (n_layers - cutoff, n_layers))

        # BatchNorm layers inside `base` stay in inference mode (moving
        # stats, not batch stats) even now, because the functional graph
        # above calls base(..., training=False) — recompiling just
        # refreshes which weights the optimizer is allowed to touch.
        train_model.compile(
            optimizer=Adam(FINE_TUNE_LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        callback2 = EarlyStopping(monitor="val_loss" if val_data else "loss", patience=8, restore_best_weights=True)
        history2 = train_model.fit(
            x_train,
            y_train,
            validation_data=val_data,
            epochs=30,
            batch_size=16,
            callbacks=[callback2],
            verbose=2,
        )
    else:
        print("Phase 2 skipped (use_fine_tune=False) — backbone stays frozen.")

    combined_history = {
        key: history1.history.get(key, []) + (history2.history.get(key, []) if history2 else [])
        for key in set(history1.history) | (set(history2.history) if history2 else set())
    }
    phase1_epochs = len(history1.history.get("loss", []))

    pred = infer_model.predict(x_test, batch_size=32, verbose=0).argmax(axis=1)
    test_acc = accuracy_score(y_test, pred)
    report = classification_report(
        y_test,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=3,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, pred, labels=list(range(len(classes))))

    print("\nTEST ACCURACY: %.3f\n" % test_acc)
    print(report)

    f_train = embed_model.predict(x_train, batch_size=32, verbose=0)
    f_val = (
        embed_model.predict(x_val, batch_size=32, verbose=0)
        if len(x_val)
        else np.zeros((0, f_train.shape[-1]), dtype="float32")
    )

    f_train_norm = _l2_normalize(f_train)
    if len(f_val):
        val_sim = (_l2_normalize(f_val) @ f_train_norm.T).max(axis=1)
        threshold = float(np.percentile(val_sim, 5))
    else:
        sim = f_train_norm @ f_train_norm.T
        np.fill_diagonal(sim, -1.0)
        threshold = float(np.percentile(sim.max(axis=1), 5))

    np.savez(
        OOD_FILE,
        feats=f_train_norm.astype("float32"),
        labels=y_train.astype("int32"),
        threshold=threshold,
    )
    print("OOD acceptance threshold (cosine):", round(threshold, 3))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    h = combined_history

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h["accuracy"], label="train")
    if "val_accuracy" in h:
        plt.plot(h["val_accuracy"], label="validation")
    plt.axvline(phase1_epochs - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune starts")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h["loss"], label="train")
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="validation")
    plt.axvline(phase1_epochs - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune starts")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "accuracy_loss.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(1.0 * len(classes) + 3, 0.9 * len(classes) + 2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (test acc = %.1f%%)" % (test_acc * 100))
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    backbone_desc = "MobileNetV2-alpha0.5"
    if use_augmentation:
        backbone_desc += ", augmented"
    if use_fine_tune:
        backbone_desc += ", top %.0f%% fine-tuned" % (FINE_TUNE_LAYER_FRACTION * 100)

    with open(os.path.join(REPORTS_DIR, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "classes": classes,
                "img_size": IMG_SIZE,
                "test_accuracy": float(test_acc),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "backbone": backbone_desc,
                "use_augmentation": use_augmentation,
                "use_fine_tune": use_fine_tune,
                "phase1_epochs": phase1_epochs,
                "phase2_epochs": len(history2.history.get("loss", [])) if history2 else 0,
                "history": h,
            },
            handle,
            indent=2,
            default=float,
        )

    with open(os.path.join(REPORTS_DIR, "metrics.md"), "w", encoding="utf-8") as handle:
        handle.write("# Evaluation results\n\n")
        handle.write("Backbone: %s.\n\n" % backbone_desc)
        handle.write("Model trained on %d classes: %s\n\n" % (len(classes), ", ".join(classes)))
        handle.write("- Train / val / test: %d / %d / %d images\n" % (len(y_train), len(y_val), len(y_test)))
        handle.write("- **Test accuracy: %.1f%%**\n\n" % (test_acc * 100))
        handle.write("```\n%s\n```\n" % report)

    infer_model.save(MODEL_DIR)
    data.save_labels(classes)

    print("Saved model -> %s/ ; labels -> %s ; reports -> %s/" % (MODEL_DIR, LABELS_FILE, REPORTS_DIR))
    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true", help="disable training-time data augmentation")
    parser.add_argument("--no-finetune", action="store_true", help="skip backbone fine-tuning (phase 2)")
    args = parser.parse_args()

    code = train_and_evaluate(use_augmentation=not args.no_augment, use_fine_tune=not args.no_finetune)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
