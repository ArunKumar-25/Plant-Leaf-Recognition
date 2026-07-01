"""Fetch a permissively-licensed image set for one species from GBIF.

Shared by reinforcement of an existing class (small --count) and growing a
new one (larger --count) -- fetches into whatever folder
data.folder_for_species() resolves to and reports via GITHUB_OUTPUT whether
that folder was new. Only CC0/CC-BY/CC-BY-SA images are kept by default
(--allow-nc opts into NC/ND). Every kept image gets an ATTRIBUTION.md entry.

Usage:
    python scripts/fetch_species_dataset.py --species "Cosmos bipinnatus" --count 30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

from plantify import config, data
from plantify.http_utils import request_with_retry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
MULTIMEDIA_EXT = "http://rs.gbif.org/terms/1.0/Multimedia"
LICENSE_FIELD = "http://purl.org/dc/terms/license"
IDENTIFIER_FIELD = "http://purl.org/dc/terms/identifier"
CREATOR_FIELD = "http://purl.org/dc/terms/creator"
RIGHTSHOLDER_FIELD = "http://purl.org/dc/terms/rightsHolder"
REFERENCES_FIELD = "http://purl.org/dc/terms/references"

PERMISSIVE_LICENSE_PREFIXES = (
    "https://creativecommons.org/publicdomain/zero",
    "https://creativecommons.org/licenses/by/",
    "https://creativecommons.org/licenses/by-sa/",
)
NC_ND_LICENSE_PREFIXES = (
    "https://creativecommons.org/licenses/by-nc-nd",
    "https://creativecommons.org/licenses/by-nc-sa",
    "https://creativecommons.org/licenses/by-nc",
    "https://creativecommons.org/licenses/by-nd",
)

MAX_SIDE = 700
QUALITY = 85
PAGE_SIZE = 100


def _license_allowed(license_url: str | None, allow_nc: bool) -> bool:
    if not license_url:
        return False
    if any(license_url.startswith(p) for p in PERMISSIVE_LICENSE_PREFIXES):
        return True
    if allow_nc and any(license_url.startswith(p) for p in NC_ND_LICENSE_PREFIXES):
        return True
    return False


def _resolve_taxon_key(species: str, timeout: int = 20) -> int | None:
    """Resolve `species` to a GBIF backbone usageKey via /species/match.

    taxonKey search matches GBIF's taxonomy hierarchy, unlike exact-string
    scientificName search -- e.g. bare genus names like "Acer" resolve to a
    valid key with real occurrences but match zero records by name alone.
    Returns None if GBIF can't resolve the name at all (matchType "NONE").
    """
    try:
        resp = request_with_retry("GET", GBIF_MATCH_URL, params={"name": species}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("GBIF species/match failed: %r", exc)
        return None
    if payload.get("matchType") in (None, "NONE"):
        return None
    return payload.get("usageKey")


def _search_media(
    species: str, allow_nc: bool, want: int, timeout: int = 20, taxon_key: int | None = None
) -> list[dict]:
    """Page through GBIF occurrence search for `species`, collecting up to
    `want` permissively-licensed media entries. If `taxon_key` is given,
    searches by GBIF's resolved taxonKey instead of free-text scientificName
    (see `_resolve_taxon_key`)."""
    found: list[dict] = []
    offset = 0
    while len(found) < want:
        params = {"mediaType": "StillImage", "limit": PAGE_SIZE, "offset": offset}
        if taxon_key is not None:
            params["taxonKey"] = taxon_key
        else:
            params["scientificName"] = species
        try:
            resp = request_with_retry(
                "GET",
                GBIF_SEARCH_URL,
                params=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("GBIF search failed: %r", exc)
            break

        results = payload.get("results", [])
        if not results:
            break

        for record in results:
            for media in record.get("extensions", {}).get(MULTIMEDIA_EXT, []):
                license_url = media.get(LICENSE_FIELD)
                if not _license_allowed(license_url, allow_nc):
                    continue
                image_url = media.get(IDENTIFIER_FIELD)
                if not image_url:
                    continue
                found.append(
                    {
                        "url": image_url,
                        "license": license_url,
                        "creator": media.get(CREATOR_FIELD) or media.get(RIGHTSHOLDER_FIELD) or "unknown",
                        "references": media.get(REFERENCES_FIELD, ""),
                        "occurrence_key": record.get("key"),
                    }
                )
                if len(found) >= want:
                    break
            if len(found) >= want:
                break

        if payload.get("endOfRecords"):
            break
        offset += PAGE_SIZE

    return found


def _download_and_downsize(url: str, dest_path: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Plantify/1.0 (+https://github.com/ArunKumar-25/Plantify)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:
        logger.warning("  download failed (%s): %r", url, exc)
        return False

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return True


def _write_attribution(dest_dir: str, species: str, entries: list[dict]) -> None:
    path = os.path.join(dest_dir, "ATTRIBUTION.md")
    is_new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if is_new_file:
            handle.write("# Image attribution\n\n")
            handle.write(
                "Images in this folder sourced from [GBIF](https://www.gbif.org/) "
                "(scientific name: %s), used under their respective licenses.\n\n" % species
            )
        for entry in entries:
            handle.write(
                "- `%s` — %s, %s. Source: <%s>\n"
                % (entry["filename"], entry["creator"], entry["license"], entry["references"] or entry["url"])
            )


def _next_candidate(path: str) -> tuple[str | None, list[str]]:
    """Read a Path-A candidate list (see new_species_candidates.txt), return
    the first not-yet-attempted species name and the full raw line list (so
    the caller can rewrite the file after a fetch attempt)."""
    if not os.path.exists(path):
        return None, []
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped, lines
    return None, lines


def _mark_candidate_done(path: str, lines: list[str], species: str) -> None:
    import datetime

    today = datetime.date.today().isoformat()
    updated = []
    marked = False
    for line in lines:
        if not marked and line.strip() == species:
            updated.append("# done: %s %s" % (today, species))
            marked = True
        else:
            updated.append(line)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(updated) + "\n")


def _base_binomial_fallback(species: str) -> str | None:
    """If `species` has more than two words (a cultivar/variety/subspecies
    qualifier, e.g. "Salix alba 'Sericea'"), return just genus + species
    epithet as a fallback search term. Returns None if there's nothing to
    fall back to (already a plain binomial or shorter)."""
    words = species.split()
    if len(words) <= 2:
        return None
    return " ".join(words[:2])


def _write_github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print("%s=%s" % (name, value))
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("%s=%s\n" % (name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Scientific name to fetch, e.g. 'Cosmos bipinnatus'")
    group.add_argument(
        "--candidate-list",
        help="Path to a Path-A candidate list file (see new_species_candidates.txt) — "
        "fetches the first unprocessed entry and marks it done on success",
    )
    parser.add_argument("--count", type=int, default=config.NEW_SPECIES_FETCH_COUNT)
    parser.add_argument("--min-fetched", type=int, default=config.NEW_SPECIES_MIN_FETCHED)
    parser.add_argument(
        "--allow-nc", action="store_true", help="Also accept CC-BY-NC/-ND licensed images (excluded by default)"
    )
    args = parser.parse_args()

    candidate_lines: list[str] = []
    if args.candidate_list:
        species, candidate_lines = _next_candidate(args.candidate_list)
        if species is None:
            logger.info("No unprocessed candidates in %s.", args.candidate_list)
            _write_github_output("fetched_count", "0")
            _write_github_output("species", "")
            return 1
    else:
        species = args.species

    folders_before = set(data.list_class_folders())
    folder = data.folder_for_species(species)
    is_new_class = folder not in folders_before
    dest_dir = os.path.join(data.DATA_DIR, folder)

    logger.info(
        "Searching GBIF for %r (allow_nc=%s, %s class)...",
        species,
        args.allow_nc,
        "new" if is_new_class else "existing",
    )
    media = _search_media(species, args.allow_nc, args.count * 2)

    # Cultivar/variety qualifiers (e.g. "Salix alba 'Sericea'") rarely match;
    # retry with just the base binomial.
    fallback = _base_binomial_fallback(species)
    if not media and fallback:
        logger.info("No results for the full name -- retrying with base binomial %r...", fallback)
        media = _search_media(fallback, args.allow_nc, args.count * 2)

    if not media:
        # Bare genus names (e.g. "Acer") don't match by scientificName text
        # search but do resolve via taxonKey -- see _resolve_taxon_key.
        taxon_key = _resolve_taxon_key(fallback or species)
        if taxon_key is not None:
            logger.info("No results via name search -- retrying via GBIF taxonKey=%d...", taxon_key)
            media = _search_media(species, args.allow_nc, args.count * 2, taxon_key=taxon_key)

    if not media:
        logger.info("No permissively-licensed media found for %r.", species)
        _write_github_output("fetched_count", "0")
        _write_github_output("species", species)
        _write_github_output("is_new_class", "true" if is_new_class else "false")
        if args.candidate_list:
            _mark_candidate_done(args.candidate_list, candidate_lines, species)
        return 1

    os.makedirs(dest_dir, exist_ok=True)
    kept = 0
    attribution_entries = []
    for entry in media:
        if kept >= args.count:
            break
        filename = "gbif_%s_%d.jpg" % (entry["occurrence_key"], kept)
        dest_path = os.path.join(dest_dir, filename)
        if _download_and_downsize(entry["url"], dest_path):
            entry["filename"] = filename
            attribution_entries.append(entry)
            kept += 1
        time.sleep(0.2)

    if attribution_entries:
        _write_attribution(dest_dir, species, attribution_entries)

    logger.info("Fetched %d image(s) for %r into %s/", kept, species, dest_dir)
    _write_github_output("fetched_count", str(kept))
    _write_github_output("species", species)
    _write_github_output("is_new_class", "true" if is_new_class else "false")
    _write_github_output("folder", folder)

    if args.candidate_list:
        _mark_candidate_done(args.candidate_list, candidate_lines, species)

    if kept < args.min_fetched:
        logger.warning(
            "Fetched fewer than --min-fetched (%d < %d) — not enough data, treating as failure.",
            kept,
            args.min_fetched,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
