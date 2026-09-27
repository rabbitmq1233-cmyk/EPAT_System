"""Trade extraction from a position series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "entry_date",
    "exit_date",
    "direction",
    "entry_price",
    "exit_price",
    "bars",
    "ret",
    "pnl",
]


def extract_trades(positions: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """Build a trade blotter from a target-position series.

    Consecutive bars sharing the same sign are collapsed into one trade.
    ``pnl`` is the signed simple return over the trade (direction * price move),
    so it is directly usable by the hit-ratio / expectancy metrics.
    """
    pos = pd.Series(positions, dtype="float64").reindex(prices.index).fillna(0.0)
    price = pd.Series(prices, dtype="float64")
    sign = np.sign(pos.to_numpy())

    records: list[dict] = []
    n = sign.size
    idx = price.index
    i = 0
    while i < n:
        direction = sign[i]
        if direction == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == direction:
            j += 1
        exit_i = min(j + 1, n - 1)
        entry_price = float(price.iloc[i])
        exit_price = float(price.iloc[exit_i])
        if entry_price <= 0:
            i = j + 1
            continue
        ret = float(direction * (exit_price / entry_price - 1.0))
        records.append(
            {
                "entry_date": idx[i],
                "exit_date": idx[exit_i],
                "direction": int(direction),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars": int(exit_i - i + 1),
                "ret": ret,
                "pnl": ret,
            }
        )
        i = j + 1

    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def extract_trades_from_returns(
    positions: pd.Series,
    strategy_returns: pd.Series,
) -> pd.DataFrame:
    """Build a trade blotter by compounding strategy returns per position segment.

    Useful for spread / portfolio strategies where there is no single traded
    price. ``pnl`` is the compounded return over the segment.
    """
    pos = pd.Series(positions, dtype="float64")
    ret = pd.Series(strategy_returns, dtype="float64").reindex(pos.index).fillna(0.0)
    sign = np.sign(pos.to_numpy())

    records: list[dict] = []
    n = sign.size
    idx = pos.index
    i = 0
    while i < n:
        direction = sign[i]
        if direction == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == direction:
            j += 1
        segment = ret.iloc[i : j + 1]
        trade_ret = float((1.0 + segment).prod() - 1.0)
        records.append(
            {
                "entry_date": idx[i],
                "exit_date": idx[min(j + 1, n - 1)],
                "direction": int(direction),
                "entry_price": float("nan"),
                "exit_price": float("nan"),
                "bars": int(j - i + 1),
                "ret": trade_ret,
                "pnl": trade_ret,
            }
        )
        i = j + 1

    return pd.DataFrame(records, columns=TRADE_COLUMNS)
