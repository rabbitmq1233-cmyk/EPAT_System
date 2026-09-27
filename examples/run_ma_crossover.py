"""End-to-end example: synthetic data -> MA crossover -> backtest -> report.

Run from the project root:

    python examples/run_ma_crossover.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epat.data import generate_ohlcv
from epat.engine import CostModel, run_vectorized
from epat.reporting import tearsheet, write_html_report, write_tearsheet
from epat.strategies import ma_crossover


def main() -> None:
    df = generate_ohlcv(n=1_000, regime=True, seed=42)
    signal = ma_crossover(df, fast=10, slow=40)

    result = run_vectorized(
        df["close"],
        signal,
        cost_model=CostModel(commission_bps=5, slippage_bps=2),
        initial_capital=100_000,
        name="ma-crossover",
    )

    print(tearsheet(result))

    out_dir = Path("reports")
    print("\ntext  ->", write_tearsheet(result, out_dir / "ma_crossover.txt"))
    print("html  ->", write_html_report(result, out_dir / "ma_crossover.html"))


if __name__ == "__main__":
    main()
