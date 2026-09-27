"""Generate a synthetic OHLCV CSV for offline experimentation.

    python examples/generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epat.data import generate_ohlcv, save_csv


def main() -> None:
    df = generate_ohlcv(n=1_500, regime=True, seed=7)
    path = save_csv(df, "data/SYNTH.csv")
    print(f"wrote {path} ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
