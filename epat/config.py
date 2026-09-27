"""Shared configuration objects and constants.

Values are intentionally simple defaults; every layer of the framework accepts
an explicit parameter so nothing is hidden in global state.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Conventional number of trading days used to annualise daily statistics.
TRADING_DAYS: int = 252

#: Default annual risk-free rate used by Sharpe/Sortino/Information ratios.
DEFAULT_RISK_FREE: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    """Economically meaningful settings for a backtest run."""

    initial_capital: float = 100_000.0
    #: Commission in basis points of traded notional (one-way).
    commission_bps: float = 0.0
    #: Slippage in basis points of price between signal and fill (one-way).
    slippage_bps: float = 0.0
    #: Annual risk-free rate used for risk-adjusted metrics.
    risk_free_rate: float = DEFAULT_RISK_FREE
    #: Annualisation factor (trading days per year).
    trading_days: int = TRADING_DAYS

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")
