"""Behavior tests for weekly candidate promotion semantics."""

from __future__ import annotations

from scripts import promote_pending


def _read_output_lines(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return dict(line.split("=", 1) for line in lines)


def test_no_pending_rows_emits_zero_counts(monkeypatch, tmp_path):
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: [])
    monkeypatch.setattr(promote_pending.data, "write_pending", lambda rows: None)

    monkeypatch.setattr("sys.argv", ["promote_pending.py"])
    assert promote_pending.main() == 0

    out = _read_output_lines(out_file)
    assert out["had_pending"] == "false"
    assert out["processed_count"] == "0"
    assert out["promoted_count"] == "0"
    assert out["new_images_count"] == "0"


def test_rejected_pending_rows_are_processed_and_persisted(monkeypatch, tmp_path):
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    rows = [
        {
            "id": "row1",
            "status": "pending",
            "plantnet_confidence": 0.10,
            "plantnet_species": "Unknown species",
            "image": "data_pending/images/row1.jpg",
        }
    ]
    written = {}

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: rows)

    def _capture_write(updated_rows):
        written["rows"] = updated_rows

    monkeypatch.setattr(promote_pending.data, "write_pending", _capture_write)

    monkeypatch.setattr("sys.argv", ["promote_pending.py"])
    assert promote_pending.main() == 0

    out = _read_output_lines(out_file)
    assert out["had_pending"] == "true"
    assert out["processed_count"] == "1"
    assert out["promoted_count"] == "0"
    assert out["new_images_count"] == "0"

    updated = written["rows"]
    assert len(updated) == 1
    assert updated[0]["status"] == "rejected"
    assert updated[0]["reject_reason"] == "below_score_threshold"


def test_promoted_rows_use_gate_pending_status(monkeypatch, tmp_path):
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"img")

    rows = [
        {
            "id": "row2",
            "status": "pending",
            "plantnet_confidence": 0.95,
            "plantnet_species": "Acer pseudoplatanus",
            "image": str(image),
        }
    ]
    written = {}

    # Real manifest rows always point inside data_pending/images/ (server-
    # generated UUID paths); point the module's trusted root at tmp_path so
    # this synthetic fixture path validates the same way a real one would.
    monkeypatch.setattr(promote_pending, "PENDING_IMAGES_ROOT", str(tmp_path))

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: rows)
    monkeypatch.setattr(
        promote_pending.data,
        "save_contribution",
        lambda src, species: ("data/leaf2/new.jpg", "leaf2"),
    )
    monkeypatch.setattr(promote_pending.data, "list_class_folders", lambda: ["leaf2"])
    monkeypatch.setattr(promote_pending.data, "species_of", lambda _folder: "Acer")

    def _capture_write(updated_rows):
        written["rows"] = updated_rows

    monkeypatch.setattr(promote_pending.data, "write_pending", _capture_write)

    monkeypatch.setattr("sys.argv", ["promote_pending.py"])
    assert promote_pending.main() == 0

    out = _read_output_lines(out_file)
    assert out["had_pending"] == "true"
    assert out["processed_count"] == "1"
    assert out["promoted_count"] == "1"
    assert out["new_images_count"] == "1"

    updated = written["rows"]
    assert len(updated) == 1
    assert updated[0]["status"] == "promoted_pending_gate"
    assert updated[0]["promoted_to"] == "data/leaf2/new.jpg"


def test_unmatched_species_stamped_with_new_species_group(monkeypatch, tmp_path):
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    rows = [
        {
            "id": "row3",
            "status": "pending",
            "plantnet_confidence": 0.90,
            "plantnet_species": "Cosmos bipinnatus",
            "image": "data_pending/images/row3.jpg",
        }
    ]
    written = {}

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: rows)
    monkeypatch.setattr(promote_pending.data, "list_class_folders", lambda: ["leaf2"])
    monkeypatch.setattr(promote_pending.data, "species_of", lambda _folder: "Acer")
    monkeypatch.setattr(promote_pending.data, "write_pending", lambda updated_rows: written.update(rows=updated_rows))

    monkeypatch.setattr("sys.argv", ["promote_pending.py"])
    assert promote_pending.main() == 0

    updated = written["rows"][0]
    assert updated["status"] == "rejected"
    assert updated["reject_reason"] == "no_matching_known_class"
    assert updated["new_species_group"] == "cosmos bipinnatus"

    out = _read_output_lines(out_file)
    assert out["new_species_ready"] == "false"


def test_invalid_image_path_rejected(monkeypatch, tmp_path):
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    rows = [
        {
            "id": "row4",
            "status": "pending",
            "plantnet_confidence": 0.95,
            "plantnet_species": "Acer pseudoplatanus",
            "image": "../../etc/passwd",  # outside data_pending/images/
        }
    ]
    written = {}

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: rows)
    monkeypatch.setattr(promote_pending.data, "list_class_folders", lambda: ["leaf2"])
    monkeypatch.setattr(promote_pending.data, "species_of", lambda _folder: "Acer")
    monkeypatch.setattr(promote_pending.data, "write_pending", lambda updated_rows: written.update(rows=updated_rows))
    save_calls = []
    monkeypatch.setattr(
        promote_pending.data, "save_contribution", lambda *a, **k: save_calls.append((a, k)) or ("x", "y")
    )

    monkeypatch.setattr("sys.argv", ["promote_pending.py"])
    assert promote_pending.main() == 0

    updated = written["rows"][0]
    assert updated["status"] == "rejected"
    assert updated["reject_reason"] == "invalid_image_path"
    assert save_calls == []  # never even attempted to copy the file


def test_check_new_species_trigger_ready_when_signals_and_diversity_met():
    rows = [
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-01T00:00:00Z"},
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-02T00:00:00Z"},
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-03T00:00:00Z"},
    ]
    ready = promote_pending._check_new_species_trigger(rows, min_signals=3, min_diversity_days=2)
    assert "cosmos bipinnatus" in ready
    assert len(ready["cosmos bipinnatus"]) == 3


def test_check_new_species_trigger_not_ready_insufficient_signals():
    rows = [
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-01T00:00:00Z"},
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-02T00:00:00Z"},
    ]
    ready = promote_pending._check_new_species_trigger(rows, min_signals=3, min_diversity_days=2)
    assert ready == {}


def test_check_new_species_trigger_not_ready_insufficient_diversity():
    rows = [
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-01T00:00:00Z"},
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-01T01:00:00Z"},
        {"new_species_group": "cosmos bipinnatus", "timestamp": "2026-01-01T02:00:00Z"},
    ]
    # 3 signals, all same calendar day -- fails the diversity bar even though
    # the count is met.
    ready = promote_pending._check_new_species_trigger(rows, min_signals=3, min_diversity_days=2)
    assert ready == {}


def test_new_species_trigger_fires_across_cycles(monkeypatch, tmp_path):
    """Two rejected rows from "previous cycles" plus one new pending row
    this cycle, same species, spread across enough distinct days, should
    fire the trigger even though only one row is newly "pending" today."""
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    rows = [
        {
            "id": "old1",
            "status": "rejected",
            "reject_reason": "no_matching_known_class",
            "new_species_group": "cosmos bipinnatus",
            "plantnet_species": "Cosmos bipinnatus",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "id": "old2",
            "status": "rejected",
            "reject_reason": "no_matching_known_class",
            "new_species_group": "cosmos bipinnatus",
            "plantnet_species": "Cosmos bipinnatus",
            "timestamp": "2026-01-02T00:00:00Z",
        },
        {
            "id": "new1",
            "status": "pending",
            "plantnet_confidence": 0.90,
            "plantnet_species": "Cosmos bipinnatus",
            "image": "data_pending/images/new1.jpg",
            "timestamp": "2026-01-03T00:00:00Z",
        },
    ]
    written = {}

    monkeypatch.setattr(promote_pending.data, "read_pending", lambda: rows)
    monkeypatch.setattr(promote_pending.data, "list_class_folders", lambda: ["leaf2"])
    monkeypatch.setattr(promote_pending.data, "species_of", lambda _folder: "Acer")
    monkeypatch.setattr(promote_pending.data, "write_pending", lambda updated_rows: written.update(rows=updated_rows))

    monkeypatch.setattr(
        "sys.argv",
        [
            "promote_pending.py",
            "--new-species-trigger-min-signals",
            "3",
            "--new-species-trigger-min-diversity-days",
            "2",
        ],
    )
    assert promote_pending.main() == 0

    out = _read_output_lines(out_file)
    assert out["new_species_ready"] == "true"
    assert out["new_species_name"] == "Cosmos bipinnatus"

    updated = {r["id"]: r for r in written["rows"]}
    assert updated["old1"]["status"] == "trigger_fired"
    assert updated["old2"]["status"] == "trigger_fired"
    assert updated["new1"]["status"] == "trigger_fired"
