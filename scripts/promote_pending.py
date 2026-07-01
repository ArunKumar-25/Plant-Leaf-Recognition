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

Also tracks a second, lighter-weight path: Pl@ntNet guesses that don't match
any existing trained class are stamped with `new_species_group` (their exact,
lowercased scientific name) instead of being discarded forever. Once enough
independent, diverse-enough guesses accumulate for the same species across
cycles, that's reported as a "new species trigger" via GITHUB_OUTPUT so the
calling workflow can fetch a real dataset for it from GBIF
(see fetch_species_dataset.py) -- this script never creates a new class
folder itself from a single unverified guess.

Usage:
    python scripts/promote_pending.py --min-plantnet-score 0.70 --max-per-class 25
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys

from plantify import config, data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PENDING_IMAGES_ROOT = os.path.abspath(data.PENDING_IMAGES_DIR)


def _match_known_species(plantnet_species: str) -> str | None:
    """Map a Pl@ntNet scientific name (e.g. "Acer pseudoplatanus") onto one
    of our existing trained classes by genus, since Pl@ntNet's binomial names
    rarely match our class labels exactly. Returns None (no match) rather than
    letting the caller silently invent a brand-new, near-empty class folder
    from an unverified external guess — the model can only ever predict
    classes that exist in class_labels.json, so growing an existing class is
    useful; fragmenting into one-off folders for "new" species is not (that's
    handled separately and much more cautiously, see _check_new_species_trigger).
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


def _is_safe_pending_image_path(src_path: str) -> bool:
    """Manifest rows are trusted only to the extent that `image` actually
    resolves inside data_pending/images/ -- a manifest crafted by hand (e.g.
    via a leaked GITHUB_CONTRIB_TOKEN or direct push access to the
    `contributions` branch) could otherwise point `image` at an arbitrary
    path on the CI runner, which would then get copied into data/ and could
    reach main via the auto-merged weekly PR with zero review."""
    if not src_path:
        return False
    resolved = os.path.abspath(src_path)
    return resolved == PENDING_IMAGES_ROOT or resolved.startswith(PENDING_IMAGES_ROOT + os.sep)


def _check_new_species_trigger(rows: list[dict], min_signals: int, min_diversity_days: int) -> dict[str, list[dict]]:
    """Group all unmatched-but-score-qualified rows (new_species_group set)
    by exact plantnet_species. Returns {group_key: [rows]} for groups that
    meet both the minimum signal count and minimum day-diversity bar --
    i.e. groups with enough independent evidence to be worth a real GBIF
    fetch this cycle. Rows already past this stage (status "trigger_fired"
    or later) are excluded by the caller before rows ever reach here.
    """
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        group = row.get("new_species_group")
        if group:
            groups[group].append(row)

    ready = {}
    for group, group_rows in groups.items():
        if len(group_rows) < min_signals:
            continue
        days = {r.get("timestamp", "")[:10] for r in group_rows}
        days.discard("")
        if len(days) < min_diversity_days:
            continue
        ready[group] = group_rows
    return ready


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
    parser.add_argument(
        "--new-species-trigger-min-signals", type=int, default=config.NEW_SPECIES_TRIGGER_MIN_SIGNALS
    )
    parser.add_argument(
        "--new-species-trigger-min-diversity-days", type=int, default=config.NEW_SPECIES_TRIGGER_MIN_DIVERSITY_DAYS
    )
    args = parser.parse_args()

    rows = data.read_pending()
    pending = [r for r in rows if r.get("status") == "pending"]

    if not pending:
        logger.info("No pending candidates — nothing to promote.")
        _write_github_output("had_pending", "false")
        _write_github_output("processed_count", "0")
        _write_github_output("promoted_count", "0")
        _write_github_output("new_images_count", "0")
        _write_github_output("new_species_ready", "false")
        _write_github_output("new_species_name", "")
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

        plantnet_species = row.get("plantnet_species", "")
        species = _match_known_species(plantnet_species)
        if species is None:
            row["status"] = "rejected"
            row["reject_reason"] = "no_matching_known_class"
            if plantnet_species.strip():
                row["new_species_group"] = plantnet_species.strip().lower()
            continue
        if per_class_promoted[species] >= args.max_per_class:
            row["status"] = "rejected"
            row["reject_reason"] = "per_class_cap_reached"
            continue
        if not _is_safe_pending_image_path(src_path):
            row["status"] = "rejected"
            row["reject_reason"] = "invalid_image_path"
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

    # Re-scan all rows carrying new_species_group (any status) so evidence
    # accumulates across weeks instead of being discarded on rejection.
    trigger_candidates = [
        r for r in rows if r.get("new_species_group") and r.get("status") not in ("trigger_fired", "accepted_committed")
    ]
    ready_groups = _check_new_species_trigger(
        trigger_candidates, args.new_species_trigger_min_signals, args.new_species_trigger_min_diversity_days
    )
    new_species_ready = False
    new_species_name = ""
    if ready_groups:
        # Process one group per cycle (keeps PR review manageable) -- pick
        # deterministically (sorted) so repeated runs behave predictably.
        group_key = sorted(ready_groups.keys())[0]
        group_rows = ready_groups[group_key]
        new_species_ready = True
        new_species_name = group_rows[0].get("plantnet_species", group_key)
        for row in group_rows:
            row["status"] = "trigger_fired"

    data.write_pending(rows)

    if promoted:
        logger.info("Promoted %d candidate(s) into data/: %s", promoted, dict(per_class_promoted))
    else:
        logger.info("No candidates met the threshold/cap this cycle.")
    if new_species_ready:
        logger.info(
            "New-species trigger ready: %r (%d signals)",
            new_species_name,
            len(ready_groups[group_key]),
        )

    _write_github_output("had_pending", "true")
    _write_github_output("processed_count", str(len(pending)))
    _write_github_output("promoted_count", str(promoted))
    _write_github_output("new_images_count", str(promoted))
    _write_github_output("new_species_ready", "true" if new_species_ready else "false")
    _write_github_output("new_species_name", new_species_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
