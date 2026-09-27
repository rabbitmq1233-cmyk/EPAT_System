"""Strategy primitives: a registry and shared helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


def close_series(data) -> pd.Series:
    """Return a float close-price series from a Series or OHLCV DataFrame."""
    if isinstance(data, pd.Series):
        return pd.Series(data, dtype="float64")
    if isinstance(data, pd.DataFrame) and "close" in data.columns:
        return pd.Series(data["close"], dtype="float64")
    raise TypeError("expected a price Series or an OHLCV DataFrame with 'close'")


@dataclass(frozen=True)
class Strategy:
    """A named signal generator: ``signal_fn(data, **params) -> target position``."""

    name: str
    signal_fn: Callable[..., pd.Series]

    def generate(self, data, **params) -> pd.Series:
        return self.signal_fn(data, **params)


#: Registry of built-in strategy factories, keyed by name.
STRATEGY_REGISTRY: dict[str, Callable[..., pd.Series]] = {}


def register(name: str) -> Callable[[Callable[..., pd.Series]], Callable[..., pd.Series]]:
    """Decorator that registers a signal function under ``name``."""

    def decorator(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        STRATEGY_REGISTRY[name] = fn
        return fn

    return decorator


def get_strategy(name: str) -> Callable[..., pd.Series]:
    """Look up a registered strategy factory by name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Names of all registered strategies."""
    return sorted(STRATEGY_REGISTRY)
