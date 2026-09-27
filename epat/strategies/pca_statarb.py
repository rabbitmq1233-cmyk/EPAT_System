"""PCA statistical arbitrage: factor-neutral residual mean reversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.indicators import zscore
from epat.strategies.base import register
from epat.strategies.mean_reversion import _reversion_state


@dataclass
class PCAStatArbResult:
    """Output of the PCA statistical-arbitrage strategy."""

    signals: pd.DataFrame
    combined_signal: pd.Series
    strategy_returns: pd.Series
    factors: pd.DataFrame
    loadings: pd.DataFrame
    residuals: pd.DataFrame


def pca_factors(
    prices: pd.DataFrame, n_components: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Principal-component factors from a price panel.

    Returns ``(factors, loadings, standardised_returns)``.
    """
    panel = pd.DataFrame(prices).dropna(how="any")
    if panel.shape[1] < 2:
        raise ValueError("PCA needs at least two assets")
    returns = panel.pct_change().dropna(how="any")
    mu = returns.mean()
    sd = returns.std(ddof=1).replace(0.0, np.nan)
    standardised = ((returns - mu) / sd).dropna(how="any")

    cov = np.cov(standardised.to_numpy(), rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components]

    factor_values = standardised.to_numpy() @ components
    factor_cols = [f"PC{i + 1}" for i in range(n_components)]
    factors = pd.DataFrame(factor_values, index=standardised.index, columns=factor_cols)
    loadings = pd.DataFrame(components, index=standardised.columns, columns=factor_cols)
    return factors, loadings, standardised


def _residualise(standardised: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    x = factors.to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    resid = {}
    for asset in standardised.columns:
        coef, *_ = np.linalg.lstsq(design, standardised[asset].to_numpy(), rcond=None)
        fitted = design @ coef
        resid[asset] = standardised[asset].to_numpy() - fitted
    return pd.DataFrame(resid, index=standardised.index)


@register("pca_statarb")
def pca_statarb(
    prices: pd.DataFrame,
    *,
    n_components: int = 1,
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    allow_short: bool = True,
) -> PCAStatArbResult:
    """Mean-revert factor-neutral residuals of a panel via PCA.

    Residuals (asset returns with the common PCA factor removed) are
    z-scored; the strategy trades the reversion to zero.
    """
    factors, loadings, standardised = pca_factors(prices, n_components)
    residuals = _residualise(standardised, factors).dropna(how="any")

    signals = {}
    strategy_returns = {}
    for asset in residuals.columns:
        z = zscore(residuals[asset], window)
        signal = _reversion_state(z, entry_z, exit_z, allow_short)
        signals[asset] = signal
        strategy_returns[asset] = signal.shift(1).fillna(0.0) * residuals[asset]

    signals_df = pd.DataFrame(signals)
    returns_df = pd.DataFrame(strategy_returns)
    combined_signal = signals_df.mean(axis=1).rename("combined_signal")
    combined_returns = returns_df.mean(axis=1).rename("strategy_returns")

    return PCAStatArbResult(
        signals=signals_df,
        combined_signal=combined_signal,
        strategy_returns=combined_returns,
        factors=factors,
        loadings=loadings,
        residuals=residuals,
    )
