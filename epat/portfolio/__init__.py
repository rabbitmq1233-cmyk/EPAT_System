"""Portfolio construction and allocation."""

from __future__ import annotations

from epat.portfolio.allocation import (
    correlation_matrix,
    efficient_frontier,
    equal_weights,
    kelly_allocation,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    random_portfolios,
)

__all__ = [
    "equal_weights",
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "correlation_matrix",
    "min_variance_weights",
    "max_sharpe_weights",
    "kelly_allocation",
    "random_portfolios",
    "efficient_frontier",
]
