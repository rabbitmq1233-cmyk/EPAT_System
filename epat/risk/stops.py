"""Stop-loss / take-profit level helpers."""

from __future__ import annotations

import pandas as pd


def atr_stop_levels(
    entry_price: float,
    atr_value: float,
    direction: int,
    sl_mult: float = 2.0,
    tp_mult: float = 3.0,
) -> tuple[float, float]:
    """Return ``(stop_price, target_price)`` for a long (``direction=1``) or
    short (``direction=-1``) trade using ATR multiples.

    A long stops *below* entry and targets *above*; a short is the reverse.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    if atr_value <= 0:
        raise ValueError("atr_value must be positive")
    stop = entry_price - direction * sl_mult * atr_value
    target = entry_price + direction * tp_mult * atr_value
    return float(stop), float(target)


def trailing_stop(
    prices: pd.Series,
    direction: int,
    atr: pd.Series | None = None,
    distance: float | None = None,
) -> pd.Series:
    """Trailing stop level for a position.

    If ``atr`` is given the trailing distance is ``distance`` multiples of ATR
    (default 3.0); otherwise ``distance`` is used as an absolute price distance.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    prices = pd.Series(prices, dtype="float64")

    if atr is not None:
        mult = 3.0 if distance is None else float(distance)
        dist = pd.Series(atr, dtype="float64") * mult
    else:
        if distance is None or distance <= 0:
            raise ValueError("provide a positive absolute distance when atr is None")
        dist = pd.Series(distance, index=prices.index, dtype="float64")

    if direction == 1:
        running = prices.cummax()
        return running - dist
    running = prices.cummin()
    return running + dist
