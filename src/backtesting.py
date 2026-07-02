"""Task 5: strategy backtesting vs. benchmark."""

from __future__ import annotations

import pandas as pd

from . import config


def run_backtest(
    returns: pd.DataFrame,
    weights: dict,
    rebalance: str | None = None,
) -> pd.Series:
    """Simulate a weighted portfolio's cumulative returns over the window.

    rebalance=None holds initial weights; "M" rebalances monthly.
    """
    raise NotImplementedError


def benchmark_returns(
    returns: pd.DataFrame,
    weights: dict = config.BENCHMARK_WEIGHTS,
) -> pd.Series:
    """Static 60/40 SPY/BND benchmark cumulative returns."""
    raise NotImplementedError


def performance_metrics(cumulative: pd.Series) -> dict:
    """Total return, annualized return, Sharpe, max drawdown."""
    raise NotImplementedError
