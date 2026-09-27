"""EPAT Trading Framework.

An end-to-end algorithmic-trading toolkit distilled from the EPAT curriculum:
data handling, indicators, strategies, a backtesting engine, risk/position
sizing, performance analytics, options pricing, and machine learning.

Import light-weight sub-packages directly, e.g.::

    from epat.data import generate_ohlcv, load_csv
    from epat.indicators import sma, rsi, atr
    from epat.strategies import ma_crossover
    from epat.engine import run_vectorized
    from epat.metrics import summarize
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "indicators",
    "metrics",
    "risk",
    "engine",
    "strategies",
    "options",
    "ml",
    "portfolio",
    "reporting",
]
