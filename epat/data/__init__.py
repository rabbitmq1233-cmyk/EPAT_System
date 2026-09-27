"""Market-data utilities: schema validation, synthetic generation, CSV loading."""

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
]
