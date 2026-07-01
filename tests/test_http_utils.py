"""Tests for the shared HTTP retry helper. No real network access."""

from __future__ import annotations

import requests

from plantify import http_utils


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_returns_immediately_on_success(monkeypatch):
    calls = []

    def _fake_request(method, url, **kwargs):
        calls.append(1)
        return _FakeResponse(200)

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _: None)

    resp = http_utils.request_with_retry("GET", "https://example.com")
    assert resp.status_code == 200
    assert len(calls) == 1


def test_returns_immediately_on_4xx_without_retrying(monkeypatch):
    calls = []

    def _fake_request(method, url, **kwargs):
        calls.append(1)
        return _FakeResponse(404)

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _: None)

    resp = http_utils.request_with_retry("GET", "https://example.com")
    assert resp.status_code == 404
    assert len(calls) == 1


def test_retries_on_connection_error_then_succeeds(monkeypatch):
    calls = []

    def _fake_request(method, url, **kwargs):
        calls.append(1)
        if len(calls) < 2:
            raise requests.ConnectionError("boom")
        return _FakeResponse(200)

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _: None)

    resp = http_utils.request_with_retry("GET", "https://example.com", retries=2)
    assert resp.status_code == 200
    assert len(calls) == 2


def test_raises_last_exception_after_exhausting_retries(monkeypatch):
    def _fake_request(method, url, **kwargs):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _: None)

    try:
        http_utils.request_with_retry("GET", "https://example.com", retries=2)
        assert False, "expected requests.Timeout to be raised"
    except requests.Timeout:
        pass


def test_retries_on_5xx_and_returns_last_response(monkeypatch):
    calls = []

    def _fake_request(method, url, **kwargs):
        calls.append(1)
        return _FakeResponse(503)

    monkeypatch.setattr(http_utils.requests, "request", _fake_request)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _: None)

    resp = http_utils.request_with_retry("GET", "https://example.com", retries=2)
    assert resp.status_code == 503
    assert len(calls) == 3
