"""Pure comparison logic for the weekly retrain regression gate. No ML, no I/O
beyond reading the two metrics JSON files produced by evaluate_model.py.

Accepts a retrained model only if:
  - overall accuracy is >= baseline accuracy - tolerance, AND
  - no class present in both baseline and new metrics regresses by more than
    `tolerance` (an aggregate-only check could hide one class getting wrecked
    while the average looks fine).

A rejected retrain is a normal, successful outcome for this script (exit 0
either way) — the workflow decides what to do with accept=true/false.

Usage:
    python scripts/regression_gate.py --baseline reports/_baseline_metrics.json \
        --new reports/_new_metrics.json --tolerance 0.01
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print(f"{name}={value}")
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def evaluate_gate(baseline: dict, new: dict, tolerance: float) -> tuple[bool, str]:
    """Pure function: returns (accept, reason). Exposed for unit testing."""
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
    for species, base_recall in baseline_per_class.items():
        new_recall = new_per_class.get(species)
        if new_recall is None:
            continue  # class not present in new eval (e.g. data churn) — not this gate's concern
        if new_recall < base_recall - tolerance:
            return False, f"per_class_regression: {species} {new_recall:.4f} < {base_recall:.4f} - {tolerance}"

    return True, "accepted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(args.new, encoding="utf-8") as handle:
        new = json.load(handle)

    accept, reason = evaluate_gate(baseline, new, args.tolerance)

    baseline_acc = baseline.get("accuracy")
    new_acc = new.get("accuracy")
    print(f"Baseline accuracy: {baseline_acc}")
    print(f"New accuracy:      {new_acc}")
    print(f"Gate result: {'ACCEPT' if accept else 'REJECT'} ({reason})")

    _write_github_output("accept", "true" if accept else "false")
    _write_github_output("reason", reason)
    _write_github_output("baseline_acc", f"{(baseline_acc or 0) * 100:.2f}")
    _write_github_output("new_acc", f"{(new_acc or 0) * 100:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
