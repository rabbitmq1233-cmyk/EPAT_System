"""Performance and risk metrics.

All functions accept pandas Series/DataFrames and are tolerant of leading
``NaN`` values. Annualisation uses ``config.TRADING_DAYS`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.config import DEFAULT_RISK_FREE, TRADING_DAYS


@dataclass
class DrawdownResult:
    """Drawdown analytics for an equity curve."""

    series: pd.Series
    max_drawdown: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None
    duration: int
    recovered: bool


def _clean(series: pd.Series) -> pd.Series:
    out = pd.Series(series, dtype="float64").dropna()
    return out


def total_return(equity: pd.Series) -> float:
    """Cumulative return of an equity curve."""
    eq = _clean(equity)
    if eq.size < 2 or eq.iloc[0] == 0:
        return float("nan")
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by an equity curve."""
    eq = _clean(equity)
    if eq.size < 2 or eq.iloc[0] <= 0:
        return float("nan")
    years = (eq.size - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    ret = _clean(returns)
    if ret.size < 2:
        return float("nan")
    return float(ret.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio."""
    ret = _clean(returns)
    if ret.size < 2:
        return float("nan")
    excess = ret - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    min_acceptable_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    ret = _clean(returns)
    if ret.size < 2:
        return float("nan")
    mar = min_acceptable_return / periods_per_year
    downside = (ret - mar).clip(upper=0.0)
    dd = np.sqrt((downside**2).mean())
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float((ret.mean() - mar) / dd * np.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> DrawdownResult:
    """Compute the full drawdown profile of an equity curve."""
    eq = _clean(equity)
    if eq.size < 2:
        empty = pd.Series(dtype="float64")
        return DrawdownResult(empty, float("nan"), None, None, None, 0, True)

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    trough_date = dd.idxmin()
    max_dd = float(dd.loc[trough_date])

    peak_date = eq.loc[:trough_date].idxmax()
    recovery_date = None
    recovered = False
    peak_level = eq.loc[peak_date]
    after = eq.loc[trough_date:]
    above = after[after >= peak_level]
    if not above.empty:
        recovery_date = above.index[0]
        recovered = True

    end_date = recovery_date if recovery_date is not None else eq.index[-1]
    duration = int(eq.index.get_loc(end_date) - eq.index.get_loc(peak_date))
    return DrawdownResult(
        series=dd,
        max_drawdown=max_dd,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        duration=duration,
        recovered=recovered,
    )


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    return drawdown(equity).max_drawdown


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    dd = abs(max_drawdown(equity))
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float(cagr(equity, periods_per_year) / dd)


def hit_ratio(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """Fraction of trades with positive P&L."""
    if trades is None or trades.empty or pnl_col not in trades.columns:
        return float("nan")
    pnl = trades[pnl_col].dropna()
    if pnl.empty:
        return float("nan")
    return float((pnl > 0).mean())


def profit_factor(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """Gross profit divided by gross loss."""
    if trades is None or trades.empty or pnl_col not in trades.columns:
        return float("nan")
    pnl = trades[pnl_col].dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def expectancy(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """Average P&L per trade."""
    if trades is None or trades.empty or pnl_col not in trades.columns:
        return float("nan")
    return float(trades[pnl_col].dropna().mean())


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Information ratio = mean active return / tracking error."""
    ret = pd.Series(returns, dtype="float64")
    bench = pd.Series(benchmark_returns, dtype="float64")
    frame = pd.concat([ret.rename("r"), bench.rename("b")], axis=1, sort=False).dropna()
    if frame.shape[0] < 2:
        return float("nan")
    active = frame["r"] - frame["b"]
    te = active.std(ddof=1)
    if te == 0 or not np.isfinite(te):
        return float("nan")
    return float(active.mean() / te * np.sqrt(periods_per_year))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free: float = DEFAULT_RISK_FREE,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    ret = pd.Series(returns, dtype="float64")
    excess = ret - risk_free / periods_per_year
    mean = excess.rolling(window, min_periods=window).mean()
    sd = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / sd.replace(0.0, np.nan)) * np.sqrt(periods_per_year)


def summarize(
    equity: pd.Series,
    returns: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    *,
    risk_free: float = DEFAULT_RISK_FREE,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Compute a standard tearsheet dictionary."""
    eq = _clean(equity)
    ret = returns if returns is not None else eq.pct_change()
    dd = drawdown(eq)
    summary = {
        "total_return": total_return(eq),
        "cagr": cagr(eq, periods_per_year),
        "annual_volatility": annual_volatility(ret, periods_per_year),
        "sharpe": sharpe_ratio(ret, risk_free, periods_per_year),
        "sortino": sortino_ratio(ret, risk_free, periods_per_year),
        "max_drawdown": dd.max_drawdown,
        "drawdown_duration": float(dd.duration),
        "calmar": calmar_ratio(eq, periods_per_year),
        "hit_ratio": hit_ratio(trades),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "num_trades": float(0 if trades is None else len(trades)),
        "final_equity": float(eq.iloc[-1]) if eq.size else float("nan"),
    }
    return summary
