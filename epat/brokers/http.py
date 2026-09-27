"""Minimal JSON HTTP client built on the standard library.

Used by the REST-based broker adapters (Zerodha, Upstox) so the framework adds
no new dependencies. The transport is injectable, which lets the adapter tests
inspect the exact requests without touching the network.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Encoding": "gzip, identity",
}


class BrokerHTTPError(RuntimeError):
    """Raised when an HTTP call to a broker fails."""

    def __init__(self, status: int, url: str, body: str = "") -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")


#: A transport takes ``(method, url, headers, body, timeout)`` and returns
#: ``(status, headers, body_bytes)``.
def default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    """Perform a real HTTP request with :mod:`urllib`."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            resp_headers = {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:  # 4xx/5xx
        payload = exc.read()
        raise BrokerHTTPError(exc.code, url, payload.decode("utf-8", "replace")) from exc
    except urllib.error.URLError as exc:
        raise BrokerHTTPError(0, url, str(exc.reason)) from exc

    if "gzip" in resp_headers.get("content-encoding", "").lower():
        raw = gzip.decompress(raw)
    return status, resp_headers, raw


def _perform(
    url: str,
    *,
    method: str,
    headers: dict[str, str] | None,
    params: dict | None,
    form: dict | None,
    timeout: float,
    retries: int,
    transport,
) -> bytes:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{query}" if query else url

    body: bytes | None = None
    request_headers = {**_HEADERS, **(headers or {})}
    if form is not None:
        body = urllib.parse.urlencode(
            {k: v for k, v in form.items() if v is not None}
        ).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            status, _resp_headers, raw = transport(method, url, request_headers, body, timeout)
            if status >= 400:
                text = raw.decode("utf-8", "replace") if raw else ""
                raise BrokerHTTPError(status, url, text)
            return raw
        except BrokerHTTPError as exc:
            last_error = exc
            if exc.status and exc.status < 500:
                raise  # client errors are not retried
            if attempt + 1 < max(1, retries):
                time.sleep(0.4 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(0.4 * (attempt + 1))
    raise BrokerHTTPError(0, url, str(last_error))


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    form: dict | None = None,
    timeout: float = 20.0,
    retries: int = 2,
    transport=default_transport,
) -> dict:
    """Perform an HTTP request and parse the JSON response body."""
    raw = _perform(
        url,
        method=method,
        headers=headers,
        params=params,
        form=form,
        timeout=timeout,
        retries=retries,
        transport=transport,
    )
    text = raw.decode("utf-8", "replace") if raw else ""
    return json.loads(text) if text else {}


def request_text(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    form: dict | None = None,
    timeout: float = 20.0,
    retries: int = 2,
    transport=default_transport,
) -> str:
    """Perform an HTTP request and return the decoded text body (e.g. CSV)."""
    raw = _perform(
        url,
        method=method,
        headers=headers,
        params=params,
        form=form,
        timeout=timeout,
        retries=retries,
        transport=transport,
    )
    return raw.decode("utf-8", "replace") if raw else ""
