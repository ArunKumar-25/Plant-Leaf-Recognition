"""Tests for the active-learning staging helpers (src/plantify/data.py's
pending-candidate functions) and the regression gate's pure comparison logic.

Fast and hermetic — no real network calls (Pl@ntNet, GitHub Contents API) and
no writes to the real data_pending/ directory (monkeypatches the module-level
path constants to a tmp_path instead).
"""

from __future__ import annotations

import os

from plantify import data
from scripts import regression_gate


def _use_tmp_pending_dir(monkeypatch, tmp_path):
    pending_dir = tmp_path / "data_pending"
    monkeypatch.setattr(data, "PENDING_DIR", str(pending_dir))
    monkeypatch.setattr(data, "PENDING_IMAGES_DIR", str(pending_dir / "images"))
    monkeypatch.setattr(data, "PENDING_MANIFEST", str(pending_dir / "manifest.jsonl"))


def test_build_pending_row_shape():
    row = data.build_pending_row(
        predicted_species="Acer",
        model_confidence=0.32,
        domain_similarity=0.55,
        decision="unknown",
        plantnet_species="Acer pseudoplatanus",
        plantnet_confidence=0.81,
        plantnet_common="Sycamore maple",
    )
    assert row["status"] == "pending"
    assert row["predicted_species"] == "Acer"
    assert row["plantnet_species"] == "Acer pseudoplatanus"
    assert row["image"].startswith("data_pending/images/")
    assert row["image"].endswith(".jpg")
    assert row["id"] in row["image"]


def test_append_and_read_pending_round_trip(monkeypatch, tmp_path):
    _use_tmp_pending_dir(monkeypatch, tmp_path)

    row1 = data.build_pending_row(
        predicted_species="Acer",
        model_confidence=0.3,
        domain_similarity=0.5,
        decision="unknown",
        plantnet_species="Acer pseudoplatanus",
        plantnet_confidence=0.8,
    )
    row2 = data.build_pending_row(
        predicted_species="Quercus",
        model_confidence=0.4,
        domain_similarity=0.52,
        decision="unknown",
        plantnet_species="Quercus robur",
        plantnet_confidence=0.9,
    )

    data.append_pending(row1)
    data.append_pending(row2)

    rows = data.read_pending()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {row1["id"], row2["id"]}


def test_read_pending_missing_file_returns_empty(monkeypatch, tmp_path):
    _use_tmp_pending_dir(monkeypatch, tmp_path)
    assert data.read_pending() == []


def test_write_pending_overwrites_with_updated_statuses(monkeypatch, tmp_path):
    _use_tmp_pending_dir(monkeypatch, tmp_path)

    row = data.build_pending_row(
        predicted_species="Tilia",
        model_confidence=0.2,
        domain_similarity=0.4,
        decision="unknown",
        plantnet_species="Tilia cordata",
        plantnet_confidence=0.75,
    )
    data.append_pending(row)

    rows = data.read_pending()
    rows[0]["status"] = "accepted_committed"
    data.write_pending(rows)

    reread = data.read_pending()
    assert len(reread) == 1
    assert reread[0]["status"] == "accepted_committed"


def test_append_pending_creates_directory(monkeypatch, tmp_path):
    _use_tmp_pending_dir(monkeypatch, tmp_path)
    assert not os.path.exists(data.PENDING_DIR)

    row = data.build_pending_row(
        predicted_species="Populus",
        model_confidence=0.1,
        domain_similarity=0.3,
        decision="unknown",
        plantnet_species="Populus tremula",
        plantnet_confidence=0.7,
    )
    data.append_pending(row)

    assert os.path.exists(data.PENDING_MANIFEST)


# --- regression_gate.py pure comparison logic ---


def test_gate_accepts_when_no_baseline():
    accept, reason = regression_gate.evaluate_gate({"accuracy": None}, {"accuracy": 0.9}, 0.01)
    assert accept is True
    assert reason == "no_baseline_to_compare"


def test_gate_rejects_when_new_has_no_test_data():
    accept, _ = regression_gate.evaluate_gate({"accuracy": 0.9}, {"accuracy": None}, 0.01)
    assert accept is False


def test_gate_accepts_equal_or_better_accuracy():
    accept, reason = regression_gate.evaluate_gate(
        {"accuracy": 0.90, "per_class": {}}, {"accuracy": 0.91, "per_class": {}}, 0.01
    )
    assert accept is True
    assert reason == "accepted"


def test_gate_rejects_aggregate_regression_beyond_tolerance():
    accept, reason = regression_gate.evaluate_gate(
        {"accuracy": 0.90, "per_class": {}}, {"accuracy": 0.85, "per_class": {}}, 0.01
    )
    assert accept is False
    assert "aggregate_regression" in reason


def test_gate_accepts_small_regression_within_tolerance():
    accept, _ = regression_gate.evaluate_gate(
        {"accuracy": 0.90, "per_class": {}}, {"accuracy": 0.895, "per_class": {}}, 0.01
    )
    assert accept is True


def test_gate_rejects_single_class_regression_even_if_aggregate_ok():
    baseline = {"accuracy": 0.90, "per_class": {"Acer": 0.95, "Quercus": 0.85}}
    # aggregate looks fine (0.90) but Quercus recall collapsed
    new = {"accuracy": 0.90, "per_class": {"Acer": 1.00, "Quercus": 0.50}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is False
    assert "per_class_regression" in reason
    assert "Quercus" in reason


def test_gate_per_class_tolerance_defaults_to_tolerance_when_not_given():
    # A single flipped image on a 15-image split swings recall ~6.7% -- with
    # no separate per_class_tolerance, that still rejects against a 1% tolerance.
    baseline = {"accuracy": 0.99, "per_class": {"Salix aurita": 1.0}}
    new = {"accuracy": 0.99, "per_class": {"Salix aurita": 14 / 15}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is False
    assert "per_class_regression" in reason


def test_gate_wider_per_class_tolerance_survives_one_flipped_image():
    baseline = {"accuracy": 0.99, "per_class": {"Salix aurita": 1.0}}
    new = {"accuracy": 0.99, "per_class": {"Salix aurita": 14 / 15}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01, per_class_tolerance=0.08)
    assert accept is True
    assert reason == "accepted"


def test_gate_wider_per_class_tolerance_still_catches_real_regression():
    baseline = {"accuracy": 0.90, "per_class": {"Acer": 0.95, "Quercus": 0.85}}
    new = {"accuracy": 0.90, "per_class": {"Acer": 1.00, "Quercus": 0.50}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01, per_class_tolerance=0.08)
    assert accept is False
    assert "per_class_regression" in reason
    assert "Quercus" in reason


def test_gate_significance_test_accepts_one_flipped_image_with_counts():
    # Same shape as the real Issue #18 rejection: 15 test images, one flip.
    # With per_class_support counts available, this is no longer statistically
    # distinguishable from noise (p > 0.05), so it should be accepted even
    # under the default (narrower) per_class_tolerance.
    baseline = {
        "accuracy": 0.99,
        "per_class": {"Acer": 1.0},
        "per_class_support": {"Acer": {"correct": 15, "total": 15}},
    }
    new = {
        "accuracy": 0.99,
        "per_class": {"Acer": 14 / 15},
        "per_class_support": {"Acer": {"correct": 14, "total": 15}},
    }
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is True
    assert reason == "accepted"


def test_gate_significance_test_still_catches_real_regression():
    baseline = {
        "accuracy": 0.90,
        "per_class": {"Quercus": 1.0},
        "per_class_support": {"Quercus": {"correct": 30, "total": 30}},
    }
    new = {
        "accuracy": 0.90,
        "per_class": {"Quercus": 20 / 30},
        "per_class_support": {"Quercus": {"correct": 20, "total": 30}},
    }
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is False
    assert "per_class_regression" in reason
    assert "Quercus" in reason


def test_gate_falls_back_to_flat_tolerance_without_support_counts():
    # No per_class_support on either side -- old behavior, unchanged.
    baseline = {"accuracy": 0.99, "per_class": {"Salix aurita": 1.0}}
    new = {"accuracy": 0.99, "per_class": {"Salix aurita": 14 / 15}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is False
    assert "per_class_regression" in reason


def test_gate_falls_back_to_flat_tolerance_when_support_missing_for_one_side():
    baseline = {
        "accuracy": 0.99,
        "per_class": {"Salix aurita": 1.0},
        "per_class_support": {"Salix aurita": {"correct": 15, "total": 15}},
    }
    new = {"accuracy": 0.99, "per_class": {"Salix aurita": 14 / 15}}  # no support on this side
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01, per_class_tolerance=0.08)
    assert accept is True
    assert reason == "accepted"


def test_gate_ignores_classes_missing_from_new_eval():
    baseline = {"accuracy": 0.90, "per_class": {"Acer": 0.95, "NewSpecies": 0.80}}
    new = {"accuracy": 0.90, "per_class": {"Acer": 0.96}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is True
    assert reason == "accepted"


def test_gate_rejects_new_species_below_recall_floor():
    baseline = {"accuracy": 0.90, "per_class": {"Acer": 0.95}}
    # "Cosmos" wasn't in baseline at all -- freshly introduced this cycle.
    new = {"accuracy": 0.90, "per_class": {"Acer": 0.96, "Cosmos": 0.33}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01, new_species_min_recall=0.60)
    assert accept is False
    assert "new_species_below_recall_floor" in reason
    assert "Cosmos" in reason


def test_gate_accepts_new_species_at_or_above_recall_floor():
    baseline = {"accuracy": 0.90, "per_class": {"Acer": 0.95}}
    new = {"accuracy": 0.90, "per_class": {"Acer": 0.96, "Cosmos": 0.67}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01, new_species_min_recall=0.60)
    assert accept is True
    assert reason == "accepted"


def test_gate_new_species_floor_defaults_without_explicit_arg():
    # Existing 3-positional-arg call sites (e.g. weekly-retrain.yml's
    # original usage pattern) must keep working unchanged.
    baseline = {"accuracy": 0.90, "per_class": {}}
    new = {"accuracy": 0.90, "per_class": {"Cosmos": 0.10}}
    accept, reason = regression_gate.evaluate_gate(baseline, new, 0.01)
    assert accept is False
    assert "new_species_below_recall_floor" in reason


def test_new_species_names_identifies_classes_absent_from_baseline():
    baseline = {"per_class": {"Acer": 0.95, "Quercus": 0.90}}
    new = {"per_class": {"Acer": 0.96, "Quercus": 0.91, "Cosmos": 0.70, "Bellis": 0.65}}
    assert regression_gate.new_species_names(baseline, new) == ["Bellis", "Cosmos"]


def test_new_species_names_empty_when_nothing_new():
    baseline = {"per_class": {"Acer": 0.95}}
    new = {"per_class": {"Acer": 0.96}}
    assert regression_gate.new_species_names(baseline, new) == []
