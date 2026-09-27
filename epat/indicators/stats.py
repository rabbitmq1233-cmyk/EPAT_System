"""Statistical indicators implemented with pure NumPy.

Includes OLS regression, the Augmented Dickey-Fuller (ADF) test, half-life of
mean reversion and Engle-Granger cointegration. Implemented directly on NumPy
so the framework has no hard dependency on statsmodels/scipy.

Note: p-values use a normal approximation, and ADF critical values are the
standard large-sample values. For publication-grade inference use statsmodels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    """Result of an ordinary-least-squares fit."""

    params: np.ndarray
    se: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    resid: np.ndarray
    rsquared: float
    nobs: int
    alpha: float
    beta: float


@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller test."""

    stat: float
    pvalue: float
    critical_values: dict[str, float]
    is_stationary: bool
    used_lag: int
    nobs: int


@dataclass
class CointegrationResult:
    """Engle-Granger cointegration result between two series."""

    hedge_ratio: float
    intercept: float
    spread: pd.Series
    adf: ADFResult
    is_cointegrated: bool


_ADF_CRITICAL = {
    "n": {"1%": -2.58, "5%": -1.95, "10%": -1.62},
    "c": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
    "ct": {"1%": -3.96, "5%": -3.41, "10%": -3.12},
}

# (statistic, p-value) anchors calibrated for the constant ("c") case.
_ADF_P_TABLE = [
    (-4.0, 0.001),
    (-3.43, 0.01),
    (-2.86, 0.05),
    (-2.57, 0.10),
    (-1.95, 0.25),
    (-1.62, 0.50),
    (0.0, 0.90),
    (1.0, 0.99),
]


def _norm_sf(z: np.ndarray | float) -> np.ndarray:
    """Standard-normal survival function via the complementary error function."""
    z = np.asarray(z, dtype="float64")
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def _lstsq_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    params, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ params
    n, k = x.shape
    ssr = float(resid @ resid)
    dof = max(n - k, 1)
    sigma2 = ssr / dof
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, params / se, np.nan)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    return params, se, tstat, resid, r2, n


def ols(y, x, add_const: bool = True) -> OLSResult:
    """Fit ``y = alpha + beta * x`` (or through the origin) by least squares."""
    yv = np.asarray(pd.Series(y), dtype="float64")
    xv = np.asarray(pd.Series(x), dtype="float64")
    mask = np.isfinite(yv) & np.isfinite(xv)
    yv, xv = yv[mask], xv[mask]
    n = yv.size
    if n < 3:
        raise ValueError("need at least 3 finite observations")
    if add_const:
        design = np.column_stack([np.ones(n), xv])
    else:
        design = xv[:, None]
    params, se, tstat, resid, r2, _ = _lstsq_stats(design, yv)
    pvalues = 2.0 * _norm_sf(np.abs(tstat))
    alpha = float(params[0]) if add_const else 0.0
    beta = float(params[1]) if add_const else float(params[0])
    return OLSResult(
        params=params,
        se=se,
        tstats=tstat,
        pvalues=pvalues,
        resid=resid,
        rsquared=float(r2),
        nobs=n,
        alpha=alpha,
        beta=beta,
    )


def _adf_lag(n: int, max_lag: int | None) -> int:
    if max_lag is not None:
        return max(0, int(max_lag))
    # Schwert (1989) rule.
    return int(np.ceil(12.0 * (n / 100.0) ** 0.25))


def _adf_pvalue(stat: float, regression: str) -> float:
    shift = {"n": -0.6, "c": 0.0, "ct": -0.5}.get(regression, 0.0)
    s = stat - shift
    if s <= _ADF_P_TABLE[0][0]:
        return _ADF_P_TABLE[0][1]
    if s >= _ADF_P_TABLE[-1][0]:
        return _ADF_P_TABLE[-1][1]
    for (x0, p0), (x1, p1) in zip(_ADF_P_TABLE, _ADF_P_TABLE[1:]):
        if x0 <= s <= x1:
            return float(p0 + (p1 - p0) * (s - x0) / (x1 - x0))
    return _ADF_P_TABLE[-1][1]


def adf(
    series,
    max_lag: int | None = None,
    regression: str = "c",
) -> ADFResult:
    """Augmented Dickey-Fuller test for a unit root.

    Null hypothesis: the series has a unit root (is non-stationary).
    Reject (series is stationary) when the t-statistic is below the 5%
    critical value.
    """
    if regression not in _ADF_CRITICAL:
        raise ValueError("regression must be one of 'n', 'c', 'ct'")

    y = np.asarray(pd.Series(series, dtype="float64").dropna(), dtype="float64")
    n = y.size
    if n < 20:
        raise ValueError("ADF test needs at least 20 observations")

    dy = np.diff(y)
    m = dy.size
    lag = _adf_lag(n, max_lag)
    lag = int(max(0, min(lag, n // 4, m - 3)))

    rows = np.arange(lag, m)
    dep = dy[rows]
    cols = [y[rows]]  # y_{t-1}
    for j in range(1, lag + 1):
        cols.append(dy[rows - j])
    design = np.column_stack(cols)

    if regression == "c":
        design = np.column_stack([np.ones(rows.size), design])
        lag_col = 1
    elif regression == "ct":
        design = np.column_stack(
            [np.ones(rows.size), np.arange(rows.size, dtype="float64"), design]
        )
        lag_col = 2
    else:
        lag_col = 0

    params, se, _, _, _, _ = _lstsq_stats(design, dep)
    stat = float(params[lag_col] / se[lag_col]) if se[lag_col] > 0 else float("nan")
    crit = _ADF_CRITICAL[regression]
    return ADFResult(
        stat=stat,
        pvalue=_adf_pvalue(stat, regression),
        critical_values=crit,
        is_stationary=bool(stat < crit["5%"]),
        used_lag=lag,
        nobs=int(rows.size),
    )


def half_life(series) -> float:
    """Half-life (in bars) of an Ornstein-Uhlenbeck mean-reverting series.

    Returns ``inf`` when the series is not mean-reverting.
    """
    s = pd.Series(series, dtype="float64").dropna()
    if s.size < 4:
        return float("inf")
    y = s.to_numpy()[1:]
    lag = s.to_numpy()[:-1]
    res = ols(y - lag, lag, add_const=True)
    lam = res.beta
    # Valid mean-reverting range is -1 < lambda < 0; anything else has no
    # finite half-life (unit root, explosive, or finite-sample artefact).
    if not np.isfinite(lam) or lam >= 0.0 or lam <= -1.0:
        return float("inf")
    return float(-math.log(2.0) / math.log(1.0 + lam))


def zscore(series, n: int = 20) -> pd.Series:
    """Rolling z-score of a series."""
    if n < 2:
        raise ValueError("n must be >= 2")
    s = pd.Series(series, dtype="float64")
    mean = s.rolling(window=n, min_periods=n).mean()
    sd = s.rolling(window=n, min_periods=n).std(ddof=0)
    return (s - mean) / sd.replace(0.0, np.nan)


#: Alias kept for readability in strategy code.
rolling_zscore = zscore


def cointegration(y, x, *, regression: str = "c", max_lag: int | None = None) -> CointegrationResult:
    """Engle-Granger two-step cointegration test between ``y`` and ``x``.

    Step 1: OLS ``y = alpha + beta*x`` (``beta`` is the hedge ratio).
    Step 2: ADF test on the regression residual (the spread).
    """
    yv = pd.Series(y, dtype="float64")
    xv = pd.Series(x, dtype="float64")
    frame = pd.concat([yv.rename("y"), xv.rename("x")], axis=1, sort=False).dropna()
    if frame.shape[0] < 20:
        raise ValueError("cointegration needs at least 20 overlapping observations")
    fit = ols(frame["y"], frame["x"], add_const=True)
    spread = pd.Series(fit.resid, index=frame.index, name="spread")
    adf_res = adf(spread, max_lag=max_lag, regression=regression)
    return CointegrationResult(
        hedge_ratio=fit.beta,
        intercept=fit.alpha,
        spread=spread,
        adf=adf_res,
        is_cointegrated=adf_res.is_stationary,
    )
