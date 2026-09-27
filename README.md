# EPAT Trading Framework

An end-to-end algorithmic-trading toolkit in Python, distilled from the EPAT
(Executive Programme in Algorithmic Trading) curriculum. It covers the full
research loop: **data → indicators → strategies → backtesting → risk/position
sizing → performance analytics**, plus **options pricing/volatility** and
**machine learning** modules.

No broker connectivity is included by design (research/backtest only). The
whole system runs offline on synthetic data, and can load your own OHLCV CSVs.

---

## Architecture

```
                        +-------------------------------+
                        |        data/                   |
                        | schema, synthetic, CSV, Yahoo |
                        +---------------+---------------+
                                        | OHLCV DataFrame
                                        v
   +---------------+        +---------------------+        +----------------+
   | indicators/   |------->|     strategies/      |------->|    engine/     |
   | SMA EMA RSI   |        | MA, breakout,        | signals| vectorized +   |
   | ATR Boll Boll  |        | mean-rev, pairs,    | (+1/-1/0) event-driven |
   | VWAP, ADF,    |        | PCA, momentum, ML    |        | costs, trades  |
   | cointegration |        +---------------------+        +--------+-------+
   +---------------+                                                  |
                                                                      v
        +---------------+        +----------------+        +------------------+
        |     risk/     |------->|    metrics/    |<-------|    reporting/    |
        | Kelly, fixed  | sizing | Sharpe Sortino |        | tearsheet, HTML  |
        | vol-target,   |        | Calmar DD      |        +------------------+
        | ATR stops     |        | expectancy     |
        +-------+-------+        +----------------+
                |                        ^
                v                        |
   +-------------------------+   +-------+-------------------------------+
   |       portfolio/        |   |              brokers/                 |
   | multi-asset pipeline,   |   | Broker ABC + mock (paper), Zerodha    |
   | efficient frontier,     |   | (Kite), Upstox, IB; execution bridge  |
   | max-Sharpe, Kelly alloc |   | trade_signals(): strategy -> orders   |
   +-------------------------+   +---------------------------------------+
```

**Data flow:** `provider → OHLCV frame → indicators → strategy signal (target
position) → risk sizing → engine (fills, costs, position state) → metrics →
report`. Multi-asset runs go through the portfolio pipeline; live/paper runs
go through the broker execution bridge.

---

## Requirements

- Python 3.10+
- `numpy`, `pandas` (only hard dependencies)

Optional extras (the framework degrades gracefully without them):
`scipy` (faster GARCH), `scikit-learn` (alternative ML estimators),
`matplotlib` (PNG/HTML equity charts), `ib_insync`/`ib_async` (Interactive
Brokers only — the Zerodha and Upstox adapters use the stdlib HTTP client).

```bash
pip install -r requirements.txt          # core
pip install -e .                         # install the `epat` package + CLI
```

Every statistics routine (OLS, ADF, cointegration, PCA, logistic regression,
GMM/EM, GARCH grid-search) is implemented on **pure NumPy**, and Yahoo Finance
data is fetched with the standard-library `urllib` — so no extra packages are
required to run.

---

## Quickstart

### CLI

```bash
python -m epat fetch --symbol RELIANCE.NS --start 2018-01-01 --out data/RELIANCE.csv
python -m epat list                         # list strategies
python -m epat demo                         # synthetic MA-crossover backtest
python -m epat backtest --symbol ^NSEI --period 5y --strategy ma_crossover --set fast=10 --set slow=40
python -m epat backtest --strategy breakout_atr --engine event --stops
python -m epat optimize --strategy ma_crossover --grid fast=5,10,20 --grid slow=40,60 --metric sharpe
python -m epat pairs --symbol HDFCBANK.NS --symbol2 ICICIBANK.NS --period 3y
python -m epat pca --symbols RELIANCE.NS,TCS.NS,INFY.NS --period 3y
python -m epat ml --symbol AAPL --period 5y --lags 5
python -m epat scan-vol --symbol ^NSEI --period 3y
python -m epat portfolio --symbols RELIANCE.NS,TCS.NS,INFY.NS --period 3y --allocation max_sharpe --sizing vol_target --rebalance 21
python -m epat paper --broker mock --strategy ma_crossover --bars 400 --capital 500000 --size 0.5
python -m epat brokers                      # list broker adapters
```

**Data source priority:** `--symbol` (Yahoo Finance) > `--data file.csv` >
synthetic generator. Yahoo flags: `--start/--end`, `--period`
(`1mo`…`5y`/`max`), `--interval` (`1d`, `1wk`, `60m`, …), `--no-adjust`
(disable split/dividend adjustment). Indian tickers use Yahoo suffixes —
`.NS` (NSE), `.BO` (BSE) — and indices are `^`-prefixed (`^NSEI` Nifty 50,
`^NSEBANK` Bank Nifty, `^BSESN` Sensex).

Common flags: `--bars`, `--seed`, `--commission-bps`, `--slippage-bps`,
`--report out.txt`, `--html out.html`.

### Python API

```python
from epat.data import generate_ohlcv
from epat.strategies import ma_crossover
from epat.engine import run_vectorized, CostModel
from epat.reporting import tearsheet

df = generate_ohlcv(n=1000, regime=True, seed=42)
signal = ma_crossover(df, fast=10, slow=40)
result = run_vectorized(df["close"], signal, cost_model=CostModel(5, 2))
print(tearsheet(result))
```

Multi-asset portfolio and paper trading:

```python
import pandas as pd
from epat.data import generate_multi
from epat.portfolio import run_portfolio_backtest
from epat.brokers import MockBroker, ExecutionConfig, get_broker, trade_signals

panel = generate_multi(n_assets=4, n=750, seed=7)
prices = pd.DataFrame({k: v["close"] for k, v in panel.items()})
signals = pd.DataFrame({k: ma_crossover(v, 10, 40) for k, v in panel.items()})

pf = run_portfolio_backtest(
    prices, signals, sizing="vol_target", allocation="max_sharpe",
    allocation_window=80, rebalance=21, cost_model=CostModel(3, 1),
)
print(pf, "\nweights:\n", pf.weights.round(3))

broker = get_broker("mock", initial_cash=500_000, commission_bps=3, slippage_bps=2)
broker.load_history("RELIANCE", panel["SYN1"])
exec_result = trade_signals(
    broker, "RELIANCE", panel["SYN1"]["close"], signals["SYN1"],
    config=ExecutionConfig(size_fraction=0.5),
)
print(exec_result.orders.tail(), exec_result.metrics["realised_pnl"])
```

More runnable examples are in `examples/`:

```bash
python examples/generate_sample_data.py
python examples/run_ma_crossover.py
python examples/run_pipeline_smoke.py      # exercises every major pipeline
python examples/run_portfolio_and_paper.py # portfolio backtest + broker paper trading
```

---

## What is implemented (and the formula)

| Area | Contents |
| --- | --- |
| `data` | schema validation, synthetic OHLCV generator, CSV loader, **Yahoo Finance fetcher** (`fetch_yahoo`, `save_yahoo_csv`, `parse_chart`; stdlib-only, split/dividend-adjusted, NSE/BSE via `.NS`/`.BO`, indices `^NSEI`/`^BSESN`) |
| `indicators` | `sma`, `ema`, `rsi(14)` (Wilder), `true_range`, `atr` (EMA form `ATR_t=(ATR_{t-1}(n-1)+TR_t)/n`), `bollinger`, `parkinson_vol` (`sqrt(mean(ln(H/L)^2)/(4ln2)·ann)`), `vwap` (`Σ(p·v)/Σv`), `zscore`, `ols`, `adf`, `half_life`, `cointegration` |
| `engine` | `run_vectorized` (position×return, one-bar lag, turnover costs) and `run_event_driven` (bar-by-bar, ATR stops, intrabar high/low fills), `CostModel`, trade blotter |
| `risk` | `kelly_binary` (`f=(pb−aq)/(ab)`), `kelly_continuous` (`f=μ/σ²`), `fractional_kelly`, `fixed_fraction_size`, `volatility_target`, `atr_position_size`, `atr_stop_levels`, `trailing_stop` |
| `metrics` | Sharpe, Sortino, Calmar, information ratio, `drawdown` (max DD, peak/trough/recovery, duration), hit-ratio, profit factor, expectancy, `summarize`, `rolling_sharpe` |
| `strategies` | `ma_crossover`, `breakout_atr` (Donchian/Turtle), `mean_reversion`, `bollinger_reversion`, `pairs` (Engle-Granger), `pca_statarb` (factor-neutral residuals), `momentum`, `roll_return_momentum`, `ml_direction` |
| `options` | Black-Scholes price, `d1_d2`, Greeks (delta/gamma/vega/theta/rho), `implied_vol` (bisection), put-call parity + synthetics, `ewma_vol`, `garch11_fit`, `realised_vol`, variance-premium stats, straddle/strangle/butterfly payoffs |
| `ml` | lagged-return features + direction labels, NumPy `LogisticRegression`, `StandardScaler`, `walk_forward_predict` (rolling/expanding OOS), `fit_gmm` (1-D Gaussian mixture via EM for volatility regimes) |
| `portfolio` | allocation library (`min_variance_weights`, `max_sharpe_weights` tangency, `kelly_allocation` `Σ⁻¹μ`, `random_portfolios`, `efficient_frontier`) **plus the multi-asset pipeline** `run_portfolio_backtest` (per-asset sizing → cost-aware net returns → weights estimated from past data only → rebalancing), `apply_sizing`, `estimate_weight_history` |
| `brokers` | `Broker` ABC + `OrderRequest`/`Order`/`Position`/`Holding`/`Margin`; **`MockBroker`** (offline paper trading: market/limit/SL fills, cash, positions, realised P&L); **`ZerodhaBroker`** (Kite Connect v3); **`UpstoxBroker`** (v2); **`IBBroker`** (ib_insync/ib_async); `get_broker`/`list_brokers`/`register_broker`; `trade_signals()` execution bridge (strategy → orders) |
| `reporting` | text tearsheet, optional HTML report with embedded equity chart |

---

## Mapping to the EPAT transcripts

| Module | Source folders | Framework area |
| --- | --- | --- |
| Portfolio & risk management (Kelly, Sharpe/Sortino, drawdown) | `10` | `risk/`, `metrics/`, `portfolio/` |
| Time-series statistics (stationarity, ADF, GMM) | `11` | `indicators/stats.py`, `ml/regimes.py` |
| Options (sinclair) + R/Quantstrat | `12 & 13` | `options/`, `engine/`, `strategies/` |
| Broker APIs, PCA stat-arb, short selling, infra | `14` | `strategies/pca_statarb.py`, `brokers/` (Zerodha/Upstox adapters) |
| Statistics & financial modelling (SFM) | `2` | `metrics/`, `portfolio/`, `indicators/` |
| Market microstructure & execution | `3` | `indicators/volume.py` (VWAP), order types + cost logic in `engine/` and `brokers/` |
| Strategy modelling & Python basics | `4 5` | `strategies/`, `engine/`, `metrics/` |
| Data modelling with Python | `6` | `data/`, `engine/` |
| Machine learning | `7` | `ml/` |
| Broker/platform automation | `9` | `brokers/` (IB adapter, execution bridge, paper trading) |

India-specific defaults that informed the design: NSE/BSE/MCX instruments,
₹ commissions via `commission_bps`, CTT/transaction costs inside `CostModel`,
and Nifty-style index work in the portfolio/stat-arb modules.

---

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

162 unit tests cover data schema, the Yahoo Finance parser/URL builder,
indicators/statistics (including ADF on white noise vs. random walk and
cointegration on a constructed pair), metrics, sizing/stops, both engines, all
strategies, options (BS vs. known values, parity, IV round-trip, GARCH), ML
(logreg, walk-forward, GMM recovery), portfolio analytics and the multi-asset
pipeline, the mock broker (fills, commissions, stops, P&L), the broker
execution bridge, and the Zerodha/Upstox request mapping (via an injected fake
HTTP transport — no network needed).

Live Yahoo Finance tests are opt-in (they require network access):

```bash
EPAT_NETWORK_TESTS=1 python -m unittest tests.test_yahoo -v
```

---

## Limitations / notes

- The **mock broker is fully tested offline**; the Zerodha/Upstox/IB adapters
  are implemented against their documented REST/API shapes and unit-tested for
  request mapping, but have **not been exercised against a live trading account**
  here (no credentials). Verify order routing with small quantities first.
- Yahoo Finance is an unofficial, rate-limited endpoint; the fetcher retries on
  transient failures but heavy use can be throttled. Cache downloads with
  `python -m epat fetch ... --out data/NAME.csv` and reuse via `--data`.
- ADF critical values / p-values use large-sample approximations; the pure-NumPy
  implementations are suitable for screening — use `statsmodels`/`arch` for
  publication-grade inference.
- GARCH(1,1) is fitted with a coarse two-stage grid search (SciPy-free).
- The mock broker does not model partial fills, margins/leverage rules, borrow
  costs, corporate actions or exchange-specific circuit/CTT rules.
- Synthetic results on random data have no edge by construction — they validate
  the plumbing, not profitability.
- This is research tooling, **not investment advice**; live trading carries risk.

### Roadmap ideas
- Intraday bars and L2 order-book / microstructure signals.
- Walk-forward parameter optimisation and combinatorial purged CV.
- Basket/OCO/GTT order helpers and a live position reconciler.
- Options strategy backtester (delta-hedged positions through the engine).
