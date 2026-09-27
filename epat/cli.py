"""Command-line interface for the EPAT trading framework.

Examples
--------
    python -m epat fetch --symbol RELIANCE.NS --start 2018-01-01 --out data/RELIANCE.csv
    python -m epat backtest --symbol ^NSEI --period 5y --strategy ma_crossover --set fast=10 --set slow=40
    python -m epat demo
    python -m epat pairs --assets 2
    python -m epat pca --assets 4
    python -m epat ml
    python -m epat scan-vol
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from epat.data import fetch_yahoo, generate_multi, generate_ohlcv, load_csv, save_csv
from epat.engine import CostModel, EventConfig, run_from_returns, run_event_driven, run_vectorized
from epat.indicators import atr
from epat.options import ewma_vol, garch11_fit, premium_stats, realised_vol
from epat.reporting import tearsheet, write_html_report, write_tearsheet
from epat.strategies import get_strategy, list_strategies, pairs, pca_statarb


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _parse_params(items: list[str]) -> dict:
    params: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --set '{item}', expected key=value")
        key, value = item.split("=", 1)
        params[key.strip()] = _coerce(value.strip())
    return params


def _load_price_data(args) -> pd.DataFrame:
    """Resolve the price data source: Yahoo symbol > CSV file > synthetic."""
    if getattr(args, "symbol", None):
        return fetch_yahoo(
            args.symbol,
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
            period=getattr(args, "period", "1y"),
            interval=getattr(args, "interval", "1d"),
            adjust=getattr(args, "adjust", True),
        )
    if getattr(args, "data", None):
        return load_csv(args.data)
    return generate_ohlcv(
        n=args.bars,
        seed=args.seed,
        mu=args.mu,
        sigma=args.sigma,
        regime=args.regime,
    )


def _cost_model(args) -> CostModel:
    return CostModel(
        commission_bps=getattr(args, "commission_bps", 0.0),
        slippage_bps=getattr(args, "slippage_bps", 0.0),
    )


def _emit(result, args) -> None:
    print(tearsheet(result))
    if getattr(args, "report", None):
        print(f"\nsaved tearsheet -> {write_tearsheet(result, args.report)}")
    if getattr(args, "html", None):
        print(f"saved html      -> {write_html_report(result, args.html)}")


def cmd_list(_args) -> int:
    import epat.ml  # noqa: F401  (registers ml_direction)

    print("Available strategies:")
    for name in list_strategies():
        print(f"  - {name}")
    return 0


def cmd_demo(args) -> int:
    df = _load_price_data(args)
    signal = get_strategy("ma_crossover")(df, fast=10, slow=40)
    result = run_vectorized(
        df["close"], signal, cost_model=_cost_model(args), name="demo-ma-crossover"
    )
    _emit(result, args)
    return 0


def cmd_backtest(args) -> int:
    import epat.ml  # noqa: F401

    df = _load_price_data(args)
    params = _parse_params(args.set)
    signal = get_strategy(args.strategy)(df, **params)

    if args.engine == "event":
        cfg = EventConfig(
            size_fraction=args.size,
            allow_short=not args.long_only,
            use_stops=args.stops,
            sl_mult=args.sl_mult,
            tp_mult=args.tp_mult,
        )
        result = run_event_driven(
            df,
            signal,
            atr=atr(df, args.atr_n) if args.stops else None,
            cost_model=_cost_model(args),
            config=cfg,
            initial_capital=args.capital,
            name=args.strategy,
        )
    else:
        result = run_vectorized(
            df["close"],
            signal,
            cost_model=_cost_model(args),
            initial_capital=args.capital,
            name=args.strategy,
        )
    _emit(result, args)
    return 0


def cmd_optimize(args) -> int:
    df = _load_price_data(args)
    grids = {}
    for item in args.grid:
        if "=" not in item:
            raise SystemExit(f"invalid --grid '{item}', expected key=v1,v2,...")
        key, values = item.split("=", 1)
        grids[key.strip()] = [_coerce(v.strip()) for v in values.split(",") if v.strip()]

    if not grids:
        raise SystemExit("provide at least one --grid key=v1,v2")

    keys = list(grids)
    rows = []
    for combo in itertools.product(*(grids[k] for k in keys)):
        params = dict(zip(keys, combo))
        try:
            signal = get_strategy(args.strategy)(df, **params)
            result = run_vectorized(
                df["close"], signal, cost_model=_cost_model(args), name=args.strategy
            )
        except Exception as exc:  # noqa: BLE001 - surface bad parameter combos
            rows.append({**params, args.metric: float("nan"), "error": str(exc)})
            continue
        rows.append({**params, args.metric: result.metrics.get(args.metric, float("nan"))})

    table = pd.DataFrame(rows).sort_values(args.metric, ascending=False)
    print(f"Optimisation grid for '{args.strategy}' ranked by {args.metric}:")
    print(table.to_string(index=False))
    return 0


def cmd_pairs(args) -> int:
    if getattr(args, "symbol", None) and getattr(args, "symbol2", None):
        common = dict(
            start=args.start,
            end=args.end,
            period=args.period,
            interval=args.interval,
            adjust=args.adjust,
        )
        y = fetch_yahoo(args.symbol, **common)["close"]
        x = fetch_yahoo(args.symbol2, **common)["close"]
    elif args.data and args.data2:
        y = load_csv(args.data)["close"]
        x = load_csv(args.data2)["close"]
    else:
        multi = generate_multi(n_assets=2, n=args.bars, seed=args.seed)
        keys = list(multi)
        y = multi[keys[0]]["close"]
        x = multi[keys[1]]["close"]

    res = pairs(y, x, train=args.train, window=args.window, entry_z=args.entry_z)
    result = run_from_returns(
        res.spread_returns, res.signal, cost_model=_cost_model(args), name="pairs"
    )
    print(f"hedge ratio = {res.hedge_ratio:.4f} | cointegrated = {res.coint.is_cointegrated}\n")
    _emit(result, args)
    return 0


def cmd_pca(args) -> int:
    symbols = getattr(args, "symbols", None)
    if symbols:
        names = [s.strip() for s in symbols.split(",") if s.strip()]
        if len(names) < 2:
            raise SystemExit("--symbols needs at least two comma-separated Yahoo symbols")
        common = dict(
            start=args.start,
            end=args.end,
            period=args.period,
            interval=args.interval,
            adjust=args.adjust,
        )
        panel = {name: fetch_yahoo(name, **common)["close"] for name in names}
        prices = pd.DataFrame(panel).dropna(how="any")
        print(f"PCA panel: {list(panel)} | {prices.shape[0]} aligned bars")
    elif args.data:
        raise SystemExit(
            "PCA needs a multi-asset panel; pass --symbols A.NS,B.NS,C.NS "
            "(Yahoo) or use --assets with synthetic data"
        )
    else:
        multi = generate_multi(n_assets=args.assets, n=args.bars, seed=args.seed)
        prices = pd.DataFrame({k: v["close"] for k, v in multi.items()})

    res = pca_statarb(prices, n_components=args.components, window=args.window, entry_z=args.entry_z)
    result = run_from_returns(
        res.strategy_returns, res.combined_signal, cost_model=_cost_model(args), name="pca-statarb"
    )
    print("PCA loadings (first component):")
    print(res.loadings.to_string())
    print()
    _emit(result, args)
    return 0


def cmd_ml(args) -> int:
    from epat.ml import ml_direction

    df = _load_price_data(args)
    signal = ml_direction(
        df, lags=args.lags, train_size=args.train_size, test_size=args.test_size
    )
    result = run_vectorized(
        df["close"], signal, cost_model=_cost_model(args), name="ml_direction"
    )
    _emit(result, args)
    return 0


def cmd_scan_vol(args) -> int:
    df = _load_price_data(args)
    returns = df["close"].pct_change().dropna()

    ewma = float(ewma_vol(returns).iloc[-1])
    print("Volatility estimators (annualised):")
    print(f"  EWMA(0.94)          {ewma:,.4f}")
    try:
        garch = garch11_fit(returns)
        print(f"  GARCH(1,1) forecast {garch.forecast_vol:,.4f}")
        print(
            f"  GARCH params        alpha={garch.alpha:.4f} beta={garch.beta:.4f} "
            f"persistence={garch.persistence:.4f}"
        )
    except ValueError as exc:
        print(f"  GARCH(1,1)          skipped ({exc})")

    if "implied_vol" in df.columns:
        iv = df["implied_vol"]
        rv = realised_vol(returns, window=args.window)
        stats = premium_stats(iv, rv)
        print("\nVariance premium (implied vs realised):")
        for key, value in stats.items():
            print(f"  {key:<20} {value:,.4f}")
    else:
        print("\n(no 'implied_vol' column found - skipping variance-premium scan)")
    return 0


def cmd_fetch(args) -> int:
    if not args.symbol:
        raise SystemExit("fetch requires --symbol (e.g. RELIANCE.NS, ^NSEI, AAPL)")
    df = fetch_yahoo(
        args.symbol,
        start=args.start,
        end=args.end,
        period=args.period,
        interval=args.interval,
        adjust=args.adjust,
    )
    print(
        f"{args.symbol}: {df.shape[0]} bars  "
        f"{df.index[0].date()} -> {df.index[-1].date()}  (interval={args.interval})"
    )
    print(df.tail(3).to_string())
    if args.out:
        print(f"\nsaved -> {save_csv(df, args.out)}")
    return 0


def _add_yahoo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--symbol",
        default=None,
        help="Yahoo Finance symbol (e.g. RELIANCE.NS, ^NSEI, AAPL); overrides --data",
    )
    parser.add_argument("--start", default=None, help="start date for --symbol (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="end date for --symbol (YYYY-MM-DD)")
    parser.add_argument(
        "--period",
        default="1y",
        help="Yahoo range when no dates given (1mo, 6mo, 1y, 5y, max)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="bar interval (1d, 1wk, 1mo, 60m, 15m, ...)",
    )
    parser.add_argument(
        "--no-adjust",
        dest="adjust",
        action="store_false",
        help="do not adjust prices for splits/dividends",
    )


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    _add_yahoo_args(parser)
    parser.add_argument("--data", type=Path, default=None, help="CSV file with OHLCV data")
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--regime", action="store_true", help="regime-switching volatility")


def _add_cost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--report", type=Path, default=None, help="write text tearsheet")
    parser.add_argument("--html", type=Path, default=None, help="write HTML report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list strategies")
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="download OHLCV from Yahoo Finance")
    _add_yahoo_args(p_fetch)
    p_fetch.add_argument("--out", type=Path, default=None, help="output CSV path")
    p_fetch.set_defaults(func=cmd_fetch)

    p_demo = sub.add_parser("demo", help="run a synthetic MA-crossover backtest")
    _add_common_data_args(p_demo)
    _add_cost_args(p_demo)
    p_demo.set_defaults(func=cmd_demo, regime=True)

    p_bt = sub.add_parser("backtest", help="backtest a registered strategy")
    _add_common_data_args(p_bt)
    _add_cost_args(p_bt)
    p_bt.add_argument("--strategy", default="ma_crossover")
    p_bt.add_argument("--set", action="append", default=[], help="strategy param key=value")
    p_bt.add_argument("--engine", choices=["vectorized", "event"], default="vectorized")
    p_bt.add_argument("--capital", type=float, default=100_000.0)
    p_bt.add_argument("--size", type=float, default=1.0, help="event engine size fraction")
    p_bt.add_argument("--long-only", action="store_true")
    p_bt.add_argument("--stops", action="store_true", help="event engine ATR stops")
    p_bt.add_argument("--sl-mult", type=float, default=2.0)
    p_bt.add_argument("--tp-mult", type=float, default=3.0)
    p_bt.add_argument("--atr-n", type=int, default=14)
    p_bt.set_defaults(func=cmd_backtest)

    p_opt = sub.add_parser("optimize", help="grid-search strategy parameters")
    _add_common_data_args(p_opt)
    _add_cost_args(p_opt)
    p_opt.add_argument("--strategy", default="ma_crossover")
    p_opt.add_argument("--grid", action="append", default=[], help="param=v1,v2,...")
    p_opt.add_argument("--metric", default="sharpe")
    p_opt.set_defaults(func=cmd_optimize)

    p_pairs = sub.add_parser("pairs", help="cointegration pairs trade")
    _add_common_data_args(p_pairs)
    _add_cost_args(p_pairs)
    p_pairs.add_argument("--data2", type=Path, default=None)
    p_pairs.add_argument("--symbol2", default=None, help="second Yahoo symbol for the pairs trade")
    p_pairs.add_argument("--train", type=int, default=120)
    p_pairs.add_argument("--window", type=int, default=60)
    p_pairs.add_argument("--entry-z", type=float, default=2.0)
    p_pairs.set_defaults(func=cmd_pairs)

    p_pca = sub.add_parser("pca", help="PCA statistical arbitrage")
    _add_common_data_args(p_pca)
    _add_cost_args(p_pca)
    p_pca.add_argument(
        "--symbols",
        default=None,
        help="comma-separated Yahoo symbols for the panel (e.g. RELIANCE.NS,TCS.NS,INFY.NS)",
    )
    p_pca.add_argument("--assets", type=int, default=4)
    p_pca.add_argument("--components", type=int, default=1)
    p_pca.add_argument("--window", type=int, default=20)
    p_pca.add_argument("--entry-z", type=float, default=1.5)
    p_pca.set_defaults(func=cmd_pca)

    p_ml = sub.add_parser("ml", help="walk-forward ML direction strategy")
    _add_common_data_args(p_ml)
    _add_cost_args(p_ml)
    p_ml.add_argument("--lags", type=int, default=12)
    p_ml.add_argument("--train-size", type=int, default=250)
    p_ml.add_argument("--test-size", type=int, default=50)
    p_ml.set_defaults(func=cmd_ml)

    p_vol = sub.add_parser("scan-vol", help="volatility estimators / variance premium")
    _add_common_data_args(p_vol)
    p_vol.add_argument("--window", type=int, default=21)
    p_vol.set_defaults(func=cmd_scan_vol)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
