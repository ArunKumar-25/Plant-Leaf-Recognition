"""Optional Pl@ntNet API client used for second-opinion predictions."""

from __future__ import annotations

import os
from typing import Dict, List

import requests

API_URL = "https://my-api.plantnet.org/v2/identify/all"

_ERRORS = {
    "no_key": "No Pl@ntNet API key configured. Add PLANTNET_API_KEY to enable this.",
    "bad_key": "Pl@ntNet rejected the API key (check it is correct).",
    "quota": "Pl@ntNet daily free quota (500/day) reached. Try again tomorrow.",
    "timeout": "Pl@ntNet took too long to respond. Try again.",
    "request_failed": "Could not reach Pl@ntNet. Check your internet connection.",
}


def get_api_key() -> str:
    try:
        import streamlit as st

        if "PLANTNET_API_KEY" in st.secrets:
            return str(st.secrets["PLANTNET_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("PLANTNET_API_KEY", "").strip()


def friendly_error(code: str) -> str:
    return _ERRORS.get(code, "Pl@ntNet request failed.")


def identify(image_path: str, organ: str = "leaf", max_results: int = 3, timeout: int = 20) -> Dict[str, List[dict] | str]:
    key = get_api_key()
    if not key:
        return {"error": "no_key"}

    try:
        with open(image_path, "rb") as handle:
            resp = requests.post(
                API_URL,
                params={
                    "api-key": key,
                    "include-related-images": "false",
                    "lang": "en",
                    "nb-results": max_results,
                },
                files={"images": handle},
                data={"organs": organ},
                timeout=timeout,
            )
        if resp.status_code in (401, 403):
            return {"error": "bad_key"}
        if resp.status_code == 429:
            return {"error": "quota"}
        if resp.status_code == 404:
            # Pl@ntNet's own "no species matched this image" response, not a
            # connectivity problem — treat it as a normal empty result.
            return {"results": []}
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        return {"error": "timeout"}
    except Exception:
        return {"error": "request_failed"}

    results = []
    for row in data.get("results", [])[:max_results]:
        species = row.get("species", {})
        results.append(
            {
                "name": species.get("scientificNameWithoutAuthor", "?"),
                "common": ", ".join(species.get("commonNames", [])[:2]),
                "score": float(row.get("score", 0.0)),
            }
        )
    return {"results": results}
