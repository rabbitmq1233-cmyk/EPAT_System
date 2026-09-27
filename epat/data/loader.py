"""CSV-based market-data loading and persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from epat.data.schema import validate_ohlcv


def load_csv(
    path: str | Path,
    *,
    date_col: str | None = None,
    require_volume: bool = False,
) -> pd.DataFrame:
    """Load a CSV file and return a validated OHLCV frame.

    ``date_col`` overrides automatic date-column detection.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    raw = pd.read_csv(path)
    if date_col is not None:
        if date_col not in raw.columns:
            raise KeyError(f"date column '{date_col}' not in {list(raw.columns)}")
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.set_index(date_col)
    return validate_ohlcv(raw, require_volume=require_volume)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write an OHLCV frame to CSV, returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = validate_ohlcv(df)
    clean.to_csv(path, index=True, index_label="date")
    return path


def list_datasets(directory: str | Path) -> list[Path]:
    """List CSV files in a directory (sorted)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))
