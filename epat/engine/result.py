"""Backtest result container."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    name: str
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        keys = ("total_return", "cagr", "sharpe", "max_drawdown", "num_trades")
        parts = ", ".join(f"{k}={self.metrics.get(k, float('nan')):.3g}" for k in keys)
        return f"BacktestResult(name={self.name!r}, {parts})"

    def summary_frame(self) -> pd.DataFrame:
        """Metrics as a two-column (metric, value) frame."""
        return pd.DataFrame(
            sorted(self.metrics.items()), columns=["metric", "value"]
        ).set_index("metric")
