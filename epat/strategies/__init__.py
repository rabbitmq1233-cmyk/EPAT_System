"""Signal-generation strategies."""

from __future__ import annotations

from epat.strategies.base import (
    STRATEGY_REGISTRY,
    Strategy,
    close_series,
    get_strategy,
    list_strategies,
    register,
)
from epat.strategies.breakout_atr import breakout_atr
from epat.strategies.ma_crossover import ma_crossover
from epat.strategies.mean_reversion import bollinger_reversion, mean_reversion
from epat.strategies.momentum import momentum, roll_return, roll_return_momentum
from epat.strategies.pairs import PairsResult, pairs
from epat.strategies.pca_statarb import PCAStatArbResult, pca_factors, pca_statarb

__all__ = [
    "Strategy",
    "STRATEGY_REGISTRY",
    "register",
    "get_strategy",
    "list_strategies",
    "close_series",
    "ma_crossover",
    "breakout_atr",
    "mean_reversion",
    "bollinger_reversion",
    "pairs",
    "PairsResult",
    "pca_factors",
    "pca_statarb",
    "PCAStatArbResult",
    "momentum",
    "roll_return",
    "roll_return_momentum",
]
