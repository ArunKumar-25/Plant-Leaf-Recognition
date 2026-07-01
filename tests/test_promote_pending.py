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
