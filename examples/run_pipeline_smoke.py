"""Run every major pipeline on synthetic data as an integration smoke test.

    python examples/run_pipeline_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from epat.data import generate_multi, generate_ohlcv
from epat.engine import CostModel, run_from_returns, run_vectorized
from epat.indicators import cointegration, half_life
from epat.ml import fit_gmm, ml_direction
from epat.options import bs_price, garch11_fit, option_greeks, premium_stats, realised_vol
from epat.strategies import ma_crossover, pairs, pca_statarb


def main() -> None:
    costs = CostModel(commission_bps=3, slippage_bps=1)
    df = generate_ohlcv(n=900, mu=0.10, sigma=0.22, regime=True, seed=1)

    r_ma = run_vectorized(df["close"], ma_crossover(df, 10, 40), cost_model=costs, name="ma")
    print("1. MA crossover backtest     :", r_ma)
    print("   half-life (bars)          :", round(half_life(df["close"]), 2))

    multi = generate_multi(n_assets=3, n=900, seed=2)
    keys = list(multi)
    pr = pairs(multi[keys[0]]["close"], multi[keys[1]]["close"])
    r_pairs = run_from_returns(pr.spread_returns, pr.signal, cost_model=costs, name="pairs")
    print("2. Pairs backtest            :", r_pairs, "| hedge", round(pr.hedge_ratio, 3))

    prices = pd.DataFrame({k: v["close"] for k, v in multi.items()})
    pca = pca_statarb(prices, n_components=1, window=20)
    r_pca = run_from_returns(pca.strategy_returns, pca.combined_signal, cost_model=costs, name="pca")
    print("3. PCA stat-arb backtest     :", r_pca)

    r_ml = run_vectorized(df["close"], ml_direction(df, lags=5, train_size=250), cost_model=costs, name="ml")
    print("4. ML direction backtest     :", r_ml)

    gmm = fit_gmm(df["close"].pct_change().dropna(), n_components=2, seed=3)
    print("5. GMM regime variances      :", [round(float(v), 6) for v in gmm.variances])

    g = garch11_fit(df["close"].pct_change().dropna(), refine=False)
    print("6. GARCH(1,1) forecast vol   :", round(g.forecast_vol, 4))

    rv = realised_vol(df["close"].pct_change().dropna(), window=21)
    stats = premium_stats((rv * 1.25).dropna(), rv)
    print("7. Variance-premium ratio    :", round(stats["premium_pct_of_iv"], 3))

    call = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    greeks = option_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    print("8. BS call / delta           :", round(call, 4), "/", round(greeks.delta, 4))

    co = cointegration(multi[keys[0]]["close"], multi[keys[1]]["close"])
    print("9. Cointegration ADF stat    :", round(co.adf.stat, 3), "| cointegrated:", co.is_cointegrated)
    print("\nPipeline smoke test complete.")


if __name__ == "__main__":
    main()
