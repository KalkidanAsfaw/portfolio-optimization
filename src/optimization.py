"""Task 4: Modern Portfolio Theory / efficient frontier."""

from __future__ import annotations

import pandas as pd


def expected_returns(returns: pd.DataFrame, tsla_forecast: float) -> pd.Series:
    """Blend forecasted TSLA return with historical BND/SPY returns."""
    raise NotImplementedError


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix of daily returns."""
    raise NotImplementedError


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame):
    """Run MPT optimization; return frontier points + key portfolios."""
    raise NotImplementedError


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame):
    """Tangency (max Sharpe) portfolio weights and metrics."""
    raise NotImplementedError


def min_volatility_portfolio(mu: pd.Series, cov: pd.DataFrame):
    """Minimum-volatility portfolio weights and metrics."""
    raise NotImplementedError
