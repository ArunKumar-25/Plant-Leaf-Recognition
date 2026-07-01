"""Shared HTTP retry helper for external API calls (Pl@ntNet, GBIF)."""

from __future__ import annotations

import time
from typing import Any

import requests


def request_with_retry(
    method: str,
    url: str,
    *,
    retries: int = 2,
    backoff: float = 0.5,
    **kwargs: Any,
) -> requests.Response:
    """requests.request(...) with backoff retry on connection errors, timeouts,
    and 5xx. 4xx returns immediately (not transient).

    Pass file bodies as bytes, not an open handle -- a handle would be
    exhausted after the first attempt and send an empty body on retry.
    """
    last_exc: Exception | None = None
    last_resp: requests.Response | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            last_resp = None
        else:
            if resp.status_code < 500:
                return resp
            last_resp = resp
            last_exc = None
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    if last_resp is not None:
        return last_resp
    assert last_exc is not None
    raise last_exc
