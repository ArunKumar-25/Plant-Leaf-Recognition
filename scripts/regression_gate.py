"""Pure comparison logic for the weekly retrain regression gate. No ML, no I/O
beyond reading the two metrics JSON files produced by evaluate_model.py.

Accepts a retrained model only if:
  - overall accuracy is >= baseline accuracy - tolerance, AND
  - no class present in both baseline and new metrics regresses by more than
    per_class_tolerance (an aggregate-only check could hide one class getting
    wrecked while the average looks fine).

per_class_tolerance defaults to `tolerance` if not given, but the two are
deliberately separate: with only ~15 test images per class, a single flipped
prediction swings that class's recall by ~6.7% -- a 1% aggregate tolerance
applied per-class would reject on pure sampling noise every time, especially
once baseline sits at (or near) 100% with no slack to absorb it. A wider
per-class tolerance (e.g. 0.08) survives one flipped image while still
catching a real multi-image regression.

A rejected retrain is a normal, successful outcome for this script (exit 0
either way) — the workflow decides what to do with accept=true/false.

Usage:
    python scripts/regression_gate.py --baseline reports/_baseline_metrics.json \
        --new reports/_new_metrics.json --tolerance 0.01 --per-class-tolerance 0.08
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print(f"{name}={value}")
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _one_sided_regression_pvalue(base_correct: int, base_total: int, new_correct: int, new_total: int) -> float | None:
    """One-sided two-proportion z-test p-value: probability of seeing a
    recall drop this large (or larger) by sampling chance alone, if the
    class's true recall hadn't actually changed. None if undefined (a zero
    total, or zero pooled variance -- e.g. both sides at 100% or both at 0%,
    where there's nothing to test)."""
    if base_total <= 0 or new_total <= 0:
        return None
    p_base = base_correct / base_total
    p_new = new_correct / new_total
    p_pool = (base_correct + new_correct) / (base_total + new_total)
    variance = p_pool * (1 - p_pool) * (1 / base_total + 1 / new_total)
    if variance <= 0:
        return None
    z = (p_base - p_new) / math.sqrt(variance)
    return 0.5 * math.erfc(z / math.sqrt(2))  # P(Z >= z), upper-tail


def evaluate_gate(
    baseline: dict,
    new: dict,
    tolerance: float,
    new_species_min_recall: float = 0.60,
    per_class_tolerance: float | None = None,
    significance: float = 0.05,
) -> tuple[bool, str]:
    """Pure function: returns (accept, reason). Exposed for unit testing."""
    if per_class_tolerance is None:
        per_class_tolerance = tolerance

    baseline_acc = baseline.get("accuracy")
    new_acc = new.get("accuracy")

    if baseline_acc is None:
        return True, "no_baseline_to_compare"
    if new_acc is None:
        return False, "new_model_has_no_test_data"

    if new_acc < baseline_acc - tolerance:
        return False, f"aggregate_regression: {new_acc:.4f} < {baseline_acc:.4f} - {tolerance}"

    baseline_per_class = baseline.get("per_class", {})
    new_per_class = new.get("per_class", {})
    baseline_support = baseline.get("per_class_support", {})
    new_support = new.get("per_class_support", {})
    for species, base_recall in baseline_per_class.items():
        new_recall = new_per_class.get(species)
        if new_recall is None:
            continue  # class not present in new eval (e.g. data churn) — not this gate's concern

        base_counts = baseline_support.get(species)
        new_counts = new_support.get(species)
        if base_counts and new_counts:
            # Prefer a significance test over counts when available: with
            # ~15 test images/class, one flipped prediction swings recall by
            # ~6.7% on pure sampling noise -- a flat percentage-point
            # tolerance can't tell that apart from a real regression, but a
            # two-proportion test can, and it scales with how much evidence
            # actually exists per class instead of a fixed number for all.
            p_value = _one_sided_regression_pvalue(
                base_counts["correct"], base_counts["total"], new_counts["correct"], new_counts["total"]
            )
            if p_value is not None:
                if p_value < significance:
                    return False, (
                        f"per_class_regression: {species} {new_recall:.4f} < {base_recall:.4f} "
                        f"(p={p_value:.4f} < {significance}, n={base_counts['total']}+{new_counts['total']})"
                    )
                continue  # drop isn't statistically significant -- not a confirmed regression

        # No usable counts for this class (older metrics files, or a caller
        # that only provides ratios, e.g. unit tests) -- fall back to the
        # flat tolerance as before.
        if new_recall < base_recall - per_class_tolerance:
            return False, (
                f"per_class_regression: {species} {new_recall:.4f} < {base_recall:.4f} - {per_class_tolerance}"
            )

    # A class with no baseline is brand new this cycle -- nothing to regress
    # against, so apply an absolute recall floor instead.
    for species, recall in new_per_class.items():
        if species in baseline_per_class:
            continue
        if recall < new_species_min_recall:
            return False, f"new_species_below_recall_floor: {species} {recall:.4f} < {new_species_min_recall}"

    return True, "accepted"


def new_species_names(baseline: dict, new: dict) -> list[str]:
    """Classes present in `new` but absent from `baseline` -- i.e. species
    the currently-deployed model has never seen, freshly introduced this
    retrain cycle. Used by the workflow to decide whether a PR needs
    mandatory human review instead of auto-merging."""
    baseline_per_class = baseline.get("per_class", {})
    new_per_class = new.get("per_class", {})
    return sorted(species for species in new_per_class if species not in baseline_per_class)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--per-class-tolerance",
        type=float,
        default=None,
        help="Per-class recall regression tolerance, wider than --tolerance by default reasoning "
        "since a small per-class test split makes even one flipped prediction swing recall a lot. "
        "Defaults to --tolerance if not given.",
    )
    parser.add_argument(
        "--new-species-min-recall",
        type=float,
        default=0.60,
        help="Absolute recall floor for any class present in --new but absent from --baseline",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="P-value threshold for the per-class two-proportion regression test, used when both "
        "metrics files carry per_class_support counts. Only rejects a class when the recall drop is "
        "this statistically unlikely to be sampling noise -- falls back to --per-class-tolerance when "
        "counts aren't available.",
    )
    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(args.new, encoding="utf-8") as handle:
        new = json.load(handle)

    accept, reason = evaluate_gate(
        baseline, new, args.tolerance, args.new_species_min_recall, args.per_class_tolerance, args.significance
    )

    baseline_acc = baseline.get("accuracy")
    new_acc = new.get("accuracy")
    logger.info("Baseline accuracy: %s", baseline_acc)
    logger.info("New accuracy:      %s", new_acc)
    logger.info("Gate result: %s (%s)", "ACCEPT" if accept else "REJECT", reason)

    names = new_species_names(baseline, new)

    _write_github_output("accept", "true" if accept else "false")
    _write_github_output("reason", reason)
    _write_github_output("baseline_acc", f"{(baseline_acc or 0) * 100:.2f}")
    _write_github_output("new_acc", f"{(new_acc or 0) * 100:.2f}")
    _write_github_output("new_species_introduced", "true" if names else "false")
    _write_github_output("new_species_names", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
