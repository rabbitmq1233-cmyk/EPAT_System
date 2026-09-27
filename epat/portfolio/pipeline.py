"""Multi-asset portfolio backtest pipeline.

Wires the whole stack together: per-asset indicators/strategies feed a risk
sizing layer, the cost-aware engine produces per-asset net returns, portfolio
weights are estimated from past data only (no look-ahead), and the result is
summarised with the metrics layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS
from epat.engine import BacktestResult, CostModel, net_returns, run_vectorized
from epat.metrics import summarize
from epat.portfolio.allocation import (
    efficient_frontier,
    equal_weights,
    kelly_allocation,
    max_sharpe_weights,
    min_variance_weights,
)
from epat.risk.sizing import volatility_target

SIZING_METHODS = ("none", "fixed", "vol_target", "kelly")
ALLOCATION_METHODS = ("equal", "inverse_vol", "min_variance", "max_sharpe", "kelly")


@dataclass
class PortfolioBacktestResult:
    """Output of :func:`run_portfolio_backtest`."""

    name: str
    equity: pd.Series
    returns: pd.Series
    per_asset: dict[str, BacktestResult]
    weights: pd.Series
    weights_history: pd.DataFrame
    correlation: pd.DataFrame
    metrics: dict[str, float]
    frontier: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        keys = ("total_return", "cagr", "sharpe", "max_drawdown")
        parts = ", ".join(f"{k}={self.metrics.get(k, float('nan')):.3g}" for k in keys)
        return f"PortfolioBacktestResult(name={self.name!r}, {parts})"


def _kelly_leverage(returns: pd.Series, window: int, max_leverage: float) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    var = returns.rolling(window, min_periods=window).var(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (mu / var).clip(lower=0.0, upper=max_leverage)
    return f.fillna(0.0)


def apply_sizing(
    prices: pd.Series,
    positions: pd.Series,
    *,
    method: str = "none",
    fraction: float = 1.0,
    target_vol: float = 0.15,
    window: int = 60,
    max_leverage: float = 2.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Scale a raw signal series by a position-sizing method.

    - ``none``       : use the raw positions
    - ``fixed``      : ``positions * fraction``
    - ``vol_target`` : target an annualised volatility (rolling realised vol)
    - ``kelly``      : rolling continuous-Kelly leverage ``mu / sigma^2`` (capped)
    """
    method = (method or "none").lower()
    if method not in SIZING_METHODS:
        raise ValueError(f"unknown sizing '{method}'; choose from {SIZING_METHODS}")

    pos = pd.Series(positions, dtype="float64").reindex(prices.index).fillna(0.0)
    if method == "none":
        return pos
    if method == "fixed":
        return pos * fraction
    returns = pd.Series(prices, dtype="float64").pct_change()
    if method == "vol_target":
        leverage = volatility_target(
            returns, target_vol=target_vol, window=window, periods_per_year=periods_per_year,
            max_leverage=max_leverage,
        )
        return pos * leverage
    return pos * _kelly_leverage(returns, window, max_leverage)


def _estimate_weights(
    history: pd.DataFrame,
    method: str,
    *,
    long_only: bool,
    risk_free: float,
) -> np.ndarray:
    n = history.shape[1]
    cov = history.cov().to_numpy()
    mu = history.mean().to_numpy() * TRADING_DAYS

    if method == "equal" or n == 1:
        weights = equal_weights(n)
    elif method == "inverse_vol":
        vol = np.sqrt(np.diag(cov))
        vol[vol == 0] = np.nan
        inv = 1.0 / vol
        weights = np.nan_to_num(inv) / np.nansum(inv) if np.nansum(inv) else equal_weights(n)
    elif method == "min_variance":
        weights = min_variance_weights(cov)
    elif method == "max_sharpe":
        weights = max_sharpe_weights(mu, cov, risk_free)
    elif method == "kelly":
        raw = kelly_allocation(mu, cov)
        total = raw.sum()
        weights = raw / total if abs(total) > 1e-12 else equal_weights(n)
    else:
        raise ValueError(f"unknown allocation '{method}'; choose from {ALLOCATION_METHODS}")

    if long_only:
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        weights = weights / total if total > 0 else equal_weights(n)
    return np.asarray(weights, dtype="float64")


def estimate_weight_history(
    net_returns_frame: pd.DataFrame,
    *,
    allocation: str = "equal",
    min_periods: int = 60,
    rebalance: int | None = None,
    long_only: bool = True,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Weights over time, estimated from past data only.

    Rebalancing uses an expanding window: at each rebalance point the weights
    are computed from returns strictly *before* that point, so the applied
    weights never use future information.
    """
    frame = pd.DataFrame(net_returns_frame).dropna(how="any")
    index = net_returns_frame.index
    columns = net_returns_frame.columns
    weights = pd.DataFrame(np.nan, index=index, columns=columns, dtype="float64")

    n = len(frame)
    if n <= min_periods:
        weights.loc[:, :] = 1.0 / len(columns)
        return weights

    step = rebalance if rebalance and rebalance > 0 else n  # single estimate by default
    starts = list(range(min_periods, n, step))
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        history = frame.iloc[:start]
        estimated = _estimate_weights(history, allocation, long_only=long_only, risk_free=risk_free)
        first = index.get_loc(frame.index[start])
        last = index.get_loc(frame.index[min(end, n - 1)])
        weights.iloc[first:last + 1] = estimated
    return weights.fillna(0.0)


def run_portfolio_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    name: str = "portfolio",
    sizing: str = "none",
    sizing_fraction: float = 1.0,
    target_vol: float = 0.15,
    sizing_window: int = 60,
    allocation: str = "equal",
    allocation_window: int = 60,
    rebalance: int | None = None,
    long_only: bool = True,
    max_leverage: float = 2.0,
    cost_model: CostModel | None = None,
    initial_capital: float = 100_000.0,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PortfolioBacktestResult:
    """Backtest a panel of strategies and combine them into a portfolio.

    Parameters
    ----------
    prices:
        Close prices per asset (columns are asset names).
    signals:
        Target position per asset, same shape as ``prices`` (values in
        ``{-1, 0, 1}`` or any fraction). Output of the strategy layer.
    sizing:
        Risk-sizing method applied per asset (see :func:`apply_sizing`).
    allocation:
        Portfolio weighting method applied to the per-asset net returns
        (``equal``, ``inverse_vol``, ``min_variance``, ``max_sharpe``, ``kelly``).
    rebalance:
        Rebalance every N bars. ``None`` estimates weights once (after
        ``allocation_window`` bars) and holds them.
    """
    price_frame = pd.DataFrame(prices).dropna(how="any")
    if price_frame.shape[1] < 1:
        raise ValueError("prices must have at least one asset")
    signal_frame = pd.DataFrame(signals).reindex(price_frame.index)[price_frame.columns].fillna(0.0)
    costs = cost_model or CostModel()

    per_asset: dict[str, BacktestResult] = {}
    net_frame = pd.DataFrame(index=price_frame.index, columns=price_frame.columns, dtype="float64")

    for column in price_frame.columns:
        sized = apply_sizing(
            price_frame[column],
            signal_frame[column],
            method=sizing,
            fraction=sizing_fraction,
            target_vol=target_vol,
            window=sizing_window,
            max_leverage=max_leverage,
            periods_per_year=periods_per_year,
        )
        per_asset[column] = run_vectorized(
            price_frame[column], sized, cost_model=costs,
            initial_capital=initial_capital, name=str(column),
        )
        net_frame[column] = net_returns(price_frame[column], sized, cost_model=costs)

    net_frame = net_frame.fillna(0.0)
    weights_history = estimate_weight_history(
        net_frame,
        allocation=allocation,
        min_periods=allocation_window,
        rebalance=rebalance,
        long_only=long_only,
        risk_free=risk_free,
    )

    gross = (weights_history * net_frame).sum(axis=1)
    turnover = weights_history.diff().abs().sum(axis=1)
    if turnover.size:
        turnover.iloc[0] = weights_history.iloc[0].abs().sum()
    portfolio_cost = turnover.fillna(0.0) * (costs.commission_rate + costs.slippage_rate)
    portfolio_returns = (gross - portfolio_cost).rename("returns")
    equity = (initial_capital * (1.0 + portfolio_returns).cumprod()).rename("equity")

    metrics = summarize(
        equity, portfolio_returns, None, risk_free=risk_free, periods_per_year=periods_per_year
    )
    metrics["num_assets"] = float(price_frame.shape[1])

    correlation = net_frame.corr()
    try:
        frontier = efficient_frontier(
            net_frame.mean().to_numpy() * periods_per_year,
            net_frame.cov().to_numpy(),
            risk_free=risk_free,
        )
    except (ValueError, np.linalg.LinAlgError):
        frontier = pd.DataFrame()

    return PortfolioBacktestResult(
        name=name,
        equity=equity,
        returns=portfolio_returns,
        per_asset=per_asset,
        weights=weights_history.iloc[-1].rename("weight"),
        weights_history=weights_history,
        correlation=correlation,
        metrics=metrics,
        frontier=frontier,
    )
