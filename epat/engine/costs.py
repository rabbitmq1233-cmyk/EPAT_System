"""Transaction-cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Commission and slippage, both expressed in basis points.

    Commission is charged on the traded notional; slippage worsens the fill
    price for a buy (higher) and a sell (lower).
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")

    @property
    def commission_rate(self) -> float:
        return self.commission_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def fill_price(self, price: float, direction: int) -> float:
        """Apply slippage to a fill price for a buy (+1) or sell (-1)."""
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        return float(price * (1.0 + direction * self.slippage_rate))

    def commission(self, notional: float) -> float:
        """Commission charged on an absolute notional traded."""
        return float(abs(notional) * self.commission_rate)
