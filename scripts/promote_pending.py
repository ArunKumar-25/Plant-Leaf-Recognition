"""Promote qualifying candidates from data_pending/ into data/<species>/.

Reads data_pending/manifest.jsonl (rows staged by api/main.py via the GitHub
Contents API, merged in from the `contributions` branch by the workflow before
this script runs). Applies the Pl@ntNet-score threshold (already applied once
at staging time in api/main.py, re-checked here defensively) and a per-class
cap so a single noisy burst can't skew one class in a given cycle. Copies
qualifying images into the curated data/<folder>/ tree, rewrites the manifest
with updated statuses (pending/rejected candidates are kept for the audit
trail, never deleted), and reports both pending and promotion counts via
GITHUB_OUTPUT for the calling workflow step.

Usage:
    python scripts/promote_pending.py --min-plantnet-score 0.70 --max-per-class 25
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plantify import data


def _match_known_species(plantnet_species: str) -> str | None:
    """Map a Pl@ntNet scientific name (e.g. "Acer pseudoplatanus") onto one
    of our existing trained classes by genus, since Pl@ntNet's binomial names
    rarely match our class labels exactly. Returns None (no match) rather than
    letting the caller silently invent a brand-new, near-empty class folder
    from an unverified external guess — the model can only ever predict
    classes that exist in class_labels.json, so growing an existing class is
    useful; fragmenting into one-off folders for "new" species is not.
    """
    genus = (
        plantnet_species.strip().split()[0].lower() if plantnet_species.strip() else ""
    )
    if not genus:
        return None
    for folder in data.list_class_folders():
        known = data.species_of(folder)
        if known.lower() == plantnet_species.strip().lower():
            return known
        if known.lower().split()[0] == genus:
            return known
    return None


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print(f"{name}={value}")
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-plantnet-score", type=float, default=0.70)
    parser.add_argument("--max-per-class", type=int, default=25)
    args = parser.parse_args()

    rows = data.read_pending()
    pending = [r for r in rows if r.get("status") == "pending"]

    if not pending:
        print("No pending candidates — nothing to promote.")
        _write_github_output("had_pending", "false")
        _write_github_output("processed_count", "0")
        _write_github_output("promoted_count", "0")
        _write_github_output("new_images_count", "0")
        return 0

    per_class_promoted: "collections.Counter[str]" = collections.Counter()
    promoted = 0

    for row in pending:
        score = row.get("plantnet_confidence", 0.0)
        src_path = row.get(
            "image", ""
        )  # relative to repo root, e.g. data_pending/images/<id>.jpg

        if score < args.min_plantnet_score:
            row["status"] = "rejected"
            row["reject_reason"] = "below_score_threshold"
            continue

        species = _match_known_species(row.get("plantnet_species", ""))
        if species is None:
            row["status"] = "rejected"
            row["reject_reason"] = "no_matching_known_class"
            continue
        if per_class_promoted[species] >= args.max_per_class:
            row["status"] = "rejected"
            row["reject_reason"] = "per_class_cap_reached"
            continue
        if not os.path.exists(src_path):
            row["status"] = "rejected"
            row["reject_reason"] = "image_missing"
            continue

        dest_path, _folder = data.save_contribution(src_path, species)
        row["status"] = "promoted_pending_gate"
        row["promoted_to"] = dest_path.replace(os.sep, "/")
        per_class_promoted[species] += 1
        promoted += 1

    data.write_pending(rows)

    if promoted:
        print(
            f"Promoted {promoted} candidate(s) into data/: {dict(per_class_promoted)}"
        )
    else:
        print("No candidates met the threshold/cap this cycle.")

    _write_github_output("had_pending", "true")
    _write_github_output("processed_count", str(len(pending)))
    _write_github_output("promoted_count", str(promoted))
    _write_github_output("new_images_count", str(promoted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
