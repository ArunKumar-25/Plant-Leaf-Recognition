"""Dataset helper utilities shared by training and inference layers."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Iterable, List, Tuple

from .config import DATA_DIR, LABELS_FILE

IMG_EXTS: Tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_PER_CLASS = 3

FOLDER_SPECIES = {
    "leaf1": "Ulmus carpinifolia",
    "leaf2": "Acer",
    "leaf3": "Salix aurita",
    "leaf4": "Quercus",
    "leaf5": "Alnus incana",
    "leaf6": "Betula pubescens",
    "leaf7": "Salix alba 'Sericea",
    "leaf8": "Populus tremula",
    "leaf9": "Ulmus glabra",
    "leaf10": "Sorbus aucuparia",
    "leaf11": "Salix sinerea",
    "leaf12": "Populus",
    "leaf13": "Tilia",
    "leaf14": "Sorbus intermedia",
    "leaf15": "Fagus silvatica",
}

DEFAULT_LABELS = ["Ulmus carpinifolia", "Acer", "Alnus incana", "Salix alba 'Sericea"]


def list_class_folders() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)))


def images_in(folder: str) -> List[str]:
    path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(path):
        return []
    return [name for name in os.listdir(path) if name.lower().endswith(IMG_EXTS)]


def species_of(folder: str) -> str:
    return FOLDER_SPECIES.get(folder, folder.replace("_", " ").strip())


def folder_for_species(species: str) -> str:
    species = species.strip()
    for folder in list_class_folders():
        if species_of(folder).lower() == species.lower():
            return folder
    slug = re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_").lower()
    return slug or "class"


def save_contribution(src_path: str, species: str) -> Tuple[str, str]:
    folder = folder_for_species(species)
    dest_dir = os.path.join(DATA_DIR, folder)
    os.makedirs(dest_dir, exist_ok=True)

    ext = os.path.splitext(src_path)[1].lower()
    if ext not in IMG_EXTS:
        ext = ".png"

    name = "user_%d%s" % (int(time.time() * 1000), ext)
    dest_path = os.path.join(dest_dir, name)
    shutil.copyfile(src_path, dest_path)
    return dest_path, folder


def save_labels(classes: Iterable[str]) -> None:
    with open(LABELS_FILE, "w", encoding="utf-8") as handle:
        json.dump(list(classes), handle, indent=2)


def load_labels(default: Iterable[str] | None = None) -> List[str]:
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, encoding="utf-8") as handle:
                labels = json.load(handle)
            if labels:
                return labels
        except Exception:
            pass
    return list(default or DEFAULT_LABELS)
