"""Yahoo Finance data provider.

Fetches OHLCV history from the public Yahoo Finance chart endpoint and returns
a validated OHLCV frame. Implemented on the standard library (``urllib`` +
``json`` + ``gzip``) so it adds **no new dependencies**.

Indian tickers use Yahoo suffixes (``.NS`` for NSE, ``.BO`` for BSE) and
indices are prefixed with ``^`` (e.g. ``^NSEI`` for Nifty 50). Convenience
constants are provided below.

Examples
--------
    from epat.data import fetch_yahoo, NIFTY
    df = fetch_yahoo("RELIANCE.NS", start="2018-01-01", interval="1d")
    nifty = fetch_yahoo(NIFTY, period="5y")
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from epat.data.loader import save_csv
from epat.data.schema import validate_ohlcv

__all__ = [
    "YahooError",
    "fetch_yahoo",
    "save_yahoo_csv",
    "NIFTY",
    "BANKNIFTY",
    "SENSEX",
    "NIFTY_MIDCAP",
    "USDINR",
]

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, identity",
}

#: Commonly used Indian index / instrument symbols on Yahoo Finance.
NIFTY = "^NSEI"
BANKNIFTY = "^NSEBANK"
SENSEX = "^BSESN"
NIFTY_MIDCAP = "^NSEMDCP50"
USDINR = "INR=X"

#: Intervals whose bars represent whole days (index is normalised to midnight).
_DAILY_INTERVALS = {"1d", "5d", "1wk", "1mo", "3mo"}

_VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}


class YahooError(RuntimeError):
    """Raised when Yahoo Finance cannot be queried or returns an error."""


def _epoch(value) -> int:
    """Convert a date-like value to a UTC epoch second (deterministic)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def build_url(
    symbol: str,
    *,
    start=None,
    end=None,
    period: str = "1y",
    interval: str = "1d",
) -> str:
    """Build the Yahoo Finance chart URL for a request."""
    if interval not in _VALID_INTERVALS:
        raise ValueError(
            f"invalid interval '{interval}'; choose from {sorted(_VALID_INTERVALS)}"
        )
    url = _BASE_URL + urllib.parse.quote(symbol, safe="")
    params: dict[str, str] = {
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    if start is not None or end is not None:
        # Yahoo rejects a request that supplies only one bound (HTTP 400),
        # so fill the missing side: start -> epoch 0, end -> now (UTC).
        start_eff = start if start is not None else 0
        end_eff = end if end is not None else pd.Timestamp.now(tz="UTC")
        params["period1"] = str(_epoch(start_eff))
        params["period2"] = str(_epoch(end_eff))
    else:
        params["range"] = period
    return f"{url}?{urllib.parse.urlencode(params)}"


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        encoding = response.headers.get("Content-Encoding", "")
    if "gzip" in encoding.lower():
        raw = gzip.decompress(raw)
    return raw


def parse_chart(
    payload: dict,
    *,
    interval: str = "1d",
    adjust: bool = True,
) -> pd.DataFrame:
    """Parse a Yahoo Finance chart JSON payload into a validated OHLCV frame.

    Pure function (no network) so it can be unit-tested offline. Rows with a
    missing close are dropped, and when ``adjust`` is set the OHLC prices are
    scaled by ``adjclose / close`` to give split/dividend-adjusted history.
    """
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "unknown error"
        raise YahooError(f"Yahoo Finance error: {description}")

    results = chart.get("result")
    if not results:
        raise YahooError("Yahoo Finance returned no result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        raise YahooError("Yahoo Finance returned an empty price series")

    quote = quotes[0]
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    n = min(len(frame), len(timestamps))
    frame = frame.iloc[:n].reset_index(drop=True)
    stamps = pd.Series(timestamps[:n])

    meta = result.get("meta") or {}
    timezone = meta.get("exchangeTimezoneName")
    index = pd.to_datetime(stamps, unit="s", utc=True)
    if timezone:
        try:
            index = index.dt.tz_convert(timezone)
        except (TypeError, ValueError):
            pass
    index = index.dt.tz_localize(None)
    if interval in _DAILY_INTERVALS:
        index = index.dt.normalize()
    frame.index = pd.DatetimeIndex(index, name="date")

    if adjust:
        adj_blocks = indicators.get("adjclose") or []
        if adj_blocks:
            adjusted = pd.Series(
                pd.to_numeric(adj_blocks[0].get("adjclose"), errors="coerce"),
                index=frame.index,
            )
            ratio = (adjusted / frame["close"]).replace([np.inf, -np.inf], np.nan)
            ratio = ratio.fillna(1.0)
            for column in ("open", "high", "low", "close"):
                frame[column] = frame[column] * ratio

    frame = frame.dropna(subset=["close"])
    if frame.empty:
        raise YahooError("Yahoo Finance returned no usable rows")
    return validate_ohlcv(frame)


def fetch_yahoo(
    symbol: str,
    *,
    start=None,
    end=None,
    period: str = "1y",
    interval: str = "1d",
    adjust: bool = True,
    timeout: float = 20.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Download OHLCV history for ``symbol`` from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Yahoo symbol, e.g. ``"AAPL"``, ``"RELIANCE.NS"``, ``"^NSEI"``.
    start, end:
        Optional date bounds. When given, ``period`` is ignored.
    period:
        Yahoo range shorthand used when no explicit dates are supplied
        (e.g. ``"1mo"``, ``"6mo"``, ``"1y"``, ``"5y"``, ``"max"``).
    interval:
        Bar size, e.g. ``"1d"``, ``"1wk"``, ``"60m"``.
    adjust:
        Scale prices for splits/dividends using the adjusted close.
    """
    url = build_url(symbol, start=start, end=end, period=period, interval=interval)
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            payload = json.loads(_http_get(url, timeout).decode("utf-8"))
            return parse_chart(payload, interval=interval, adjust=adjust)
        except YahooError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(0.5 * (attempt + 1))
    raise YahooError(f"failed to fetch '{symbol}' from Yahoo Finance: {last_error}")


def save_yahoo_csv(
    symbol: str,
    path: str | Path,
    **kwargs,
) -> Path:
    """Fetch ``symbol`` from Yahoo Finance and save it as a CSV."""
    frame = fetch_yahoo(symbol, **kwargs)
    return save_csv(frame, path)
