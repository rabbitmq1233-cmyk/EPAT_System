"""Market-data utilities: schema validation, synthetic generation, CSV loading,
and Yahoo Finance fetching."""

from __future__ import annotations

from epat.data.loader import list_datasets, load_csv, save_csv
from epat.data.schema import (
    OHLCV_COLUMNS,
    DataSchemaError,
    log_returns,
    simple_returns,
    validate_ohlcv,
)
from epat.data.synthetic import generate_multi, generate_ohlcv
from epat.data.yahoo import (
    BANKNIFTY,
    NIFTY,
    NIFTY_MIDCAP,
    SENSEX,
    USDINR,
    YahooError,
    build_url,
    fetch_yahoo,
    parse_chart,
    save_yahoo_csv,
)

__all__ = [
    "OHLCV_COLUMNS",
    "DataSchemaError",
    "log_returns",
    "simple_returns",
    "validate_ohlcv",
    "generate_ohlcv",
    "generate_multi",
    "load_csv",
    "save_csv",
    "list_datasets",
    "fetch_yahoo",
    "save_yahoo_csv",
    "parse_chart",
    "build_url",
    "YahooError",
    "NIFTY",
    "BANKNIFTY",
    "SENSEX",
    "NIFTY_MIDCAP",
    "USDINR",
]
