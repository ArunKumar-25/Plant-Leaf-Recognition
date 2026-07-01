"""Tests for the GBIF species-dataset fetcher. No real network access —
GBIF search and image downloads are mocked throughout.
"""

from __future__ import annotations

from plantify import http_utils
from scripts import fetch_species_dataset as fsd


def test_base_binomial_fallback_strips_cultivar_qualifier():
    assert fsd._base_binomial_fallback("Salix alba 'Sericea") == "Salix alba"
    assert fsd._base_binomial_fallback("Salix alba var. sericea") == "Salix alba"


def test_base_binomial_fallback_none_for_plain_binomial():
    assert fsd._base_binomial_fallback("Cosmos bipinnatus") is None
    assert fsd._base_binomial_fallback("Acer") is None


def test_resolve_taxon_key_returns_usage_key_on_confident_match(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"usageKey": 3189834, "matchType": "HIGHERRANK", "confidence": 92}

    monkeypatch.setattr(http_utils.requests, "request", lambda *a, **k: _FakeResponse())
    assert fsd._resolve_taxon_key("Acer") == 3189834


def test_resolve_taxon_key_returns_none_when_ambiguous(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"matchType": "NONE", "note": "Multiple equal matches for Salix alba"}

    monkeypatch.setattr(http_utils.requests, "request", lambda *a, **k: _FakeResponse())
    assert fsd._resolve_taxon_key("Salix alba") is None


def test_resolve_taxon_key_returns_none_on_request_failure(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("boom")

    monkeypatch.setattr(http_utils.requests, "request", lambda *a, **k: _raise())
    assert fsd._resolve_taxon_key("Acer") is None


def test_search_media_uses_taxon_key_param_when_given(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"endOfRecords": True, "results": []}

    captured = {}

    def _fake_request(method, url, params, timeout):
        captured.update(params)
        return _FakeResponse()

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    fsd._search_media("Acer", allow_nc=False, want=5, taxon_key=3189834)

    assert captured["taxonKey"] == 3189834
    assert "scientificName" not in captured


def test_license_allowed_permissive_by_default():
    assert fsd._license_allowed("https://creativecommons.org/publicdomain/zero/1.0/", allow_nc=False)
    assert fsd._license_allowed("https://creativecommons.org/licenses/by/4.0/", allow_nc=False)
    assert fsd._license_allowed("https://creativecommons.org/licenses/by-sa/4.0/", allow_nc=False)


def test_license_rejects_nc_and_nd_by_default():
    assert not fsd._license_allowed("https://creativecommons.org/licenses/by-nc/4.0/", allow_nc=False)
    assert not fsd._license_allowed("https://creativecommons.org/licenses/by-nd/4.0/", allow_nc=False)
    assert not fsd._license_allowed("https://creativecommons.org/licenses/by-nc-sa/4.0/", allow_nc=False)


def test_license_allows_nc_when_opted_in():
    assert fsd._license_allowed("https://creativecommons.org/licenses/by-nc/4.0/", allow_nc=True)


def test_license_rejects_missing_or_unrecognized():
    assert not fsd._license_allowed(None, allow_nc=False)
    assert not fsd._license_allowed("", allow_nc=False)
    assert not fsd._license_allowed("all-rights-reserved", allow_nc=False)


def test_search_media_filters_by_license(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    payload = {
        "endOfRecords": True,
        "results": [
            {
                "key": 111,
                "extensions": {
                    fsd.MULTIMEDIA_EXT: [
                        {
                            fsd.LICENSE_FIELD: "https://creativecommons.org/licenses/by/4.0/",
                            fsd.IDENTIFIER_FIELD: "https://example.com/allowed.jpg",
                            fsd.CREATOR_FIELD: "alice",
                            fsd.REFERENCES_FIELD: "https://example.com/obs/111",
                        }
                    ]
                },
            },
            {
                "key": 222,
                "extensions": {
                    fsd.MULTIMEDIA_EXT: [
                        {
                            fsd.LICENSE_FIELD: "https://creativecommons.org/licenses/by-nc/4.0/",
                            fsd.IDENTIFIER_FIELD: "https://example.com/blocked.jpg",
                            fsd.CREATOR_FIELD: "bob",
                        }
                    ]
                },
            },
        ],
    }

    monkeypatch.setattr(http_utils.requests, "request", lambda *a, **k: _FakeResponse(payload))

    media = fsd._search_media("Cosmos bipinnatus", allow_nc=False, want=10)
    assert len(media) == 1
    assert media[0]["url"] == "https://example.com/allowed.jpg"
    assert media[0]["creator"] == "alice"


def test_write_attribution_creates_file_with_entries(tmp_path):
    dest_dir = tmp_path / "cosmos_bipinnatus"
    dest_dir.mkdir()
    entries = [
        {
            "filename": "gbif_111_0.jpg",
            "creator": "alice",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "references": "https://example.com/obs/111",
            "url": "https://example.com/allowed.jpg",
        }
    ]
    fsd._write_attribution(str(dest_dir), "Cosmos bipinnatus", entries)

    path = dest_dir / "ATTRIBUTION.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "gbif_111_0.jpg" in content
    assert "alice" in content
    assert "Cosmos bipinnatus" in content


def test_write_attribution_appends_without_duplicating_header(tmp_path):
    dest_dir = tmp_path / "cosmos_bipinnatus"
    dest_dir.mkdir()
    entry = {
        "filename": "a.jpg",
        "creator": "alice",
        "license": "CC-BY",
        "references": "",
        "url": "https://example.com/a.jpg",
    }
    fsd._write_attribution(str(dest_dir), "Cosmos bipinnatus", [entry])
    fsd._write_attribution(str(dest_dir), "Cosmos bipinnatus", [{**entry, "filename": "b.jpg"}])

    content = (dest_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert content.count("# Image attribution") == 1
    assert "a.jpg" in content
    assert "b.jpg" in content


def test_next_candidate_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "candidates.txt"
    path.write_text("# a comment\n\nTaraxacum officinale\nBellis perennis\n", encoding="utf-8")

    species, lines = fsd._next_candidate(str(path))
    assert species == "Taraxacum officinale"
    assert "Bellis perennis" in lines


def test_next_candidate_returns_none_when_all_done(tmp_path):
    path = tmp_path / "candidates.txt"
    path.write_text("# done: 2026-01-01 Taraxacum officinale\n", encoding="utf-8")

    species, _lines = fsd._next_candidate(str(path))
    assert species is None


def test_next_candidate_missing_file_returns_none(tmp_path):
    species, lines = fsd._next_candidate(str(tmp_path / "does_not_exist.txt"))
    assert species is None
    assert lines == []


def test_mark_candidate_done_rewrites_only_matching_line(tmp_path):
    path = tmp_path / "candidates.txt"
    lines = ["Taraxacum officinale", "Bellis perennis"]
    fsd._mark_candidate_done(str(path), lines, "Taraxacum officinale")

    content = path.read_text(encoding="utf-8")
    assert "# done:" in content
    assert "Taraxacum officinale" in content
    assert "Bellis perennis" in content
    # the untouched line stays a plain, retryable entry
    assert any(line.strip() == "Bellis perennis" for line in content.splitlines())
