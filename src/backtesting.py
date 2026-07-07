"""Task 5: strategy backtesting vs. benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def run_backtest(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
    rebalance: str | None = None,
) -> pd.Series:
    """Simulate a weighted portfolio's cumulative value (start = 1.0).

    ``rebalance=None`` buys and holds: each asset's initial dollar slice
    compounds independently (weights drift with performance). A pandas
    frequency like ``"ME"`` (month-end) resets to target weights at each
    period boundary.
    """
    w = pd.Series(weights, dtype=float)
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise KeyError(f"Weights reference assets missing from returns: {missing}")
    rets = returns[w.index].dropna(how="any")

    if rebalance is None:
        # Buy & hold: portfolio value = sum of independently compounding slices.
        growth = (1 + rets).cumprod()
        value = growth.mul(w, axis=1).sum(axis=1)
        return value.rename("portfolio_value")

    value_segments = []
    start_value = 1.0
    for _, seg in rets.groupby(pd.Grouper(freq=rebalance)):
        if seg.empty:
            continue
        growth = (1 + seg).cumprod()
        seg_value = growth.mul(w * start_value, axis=1).sum(axis=1)
        value_segments.append(seg_value)
        start_value = float(seg_value.iloc[-1])
    return pd.concat(value_segments).rename("portfolio_value")


def benchmark_portfolio(
    returns: pd.DataFrame,
    weights: dict | None = None,
) -> pd.Series:
    """Static 60/40 SPY/BND benchmark cumulative value (buy & hold)."""
    return run_backtest(returns, weights or config.BENCHMARK_WEIGHTS).rename(
        "benchmark_value"
    )


def max_drawdown(value: pd.Series) -> float:
    """Largest peak-to-trough decline of a value curve (negative number)."""
    running_max = value.cummax()
    drawdown = value / running_max - 1.0
    return float(drawdown.min())


def performance_metrics(
    value: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    periods: int = config.TRADING_DAYS,
) -> dict:
    """Total return, annualized return, Sharpe ratio, and max drawdown."""
    daily = value.pct_change().dropna()
    total_return = float(value.iloc[-1] / value.iloc[0] - 1.0)
    n = len(daily)
    annual_return = float((1 + total_return) ** (periods / n) - 1) if n else np.nan
    vol = float(daily.std() * np.sqrt(periods))
    sharpe = (annual_return - risk_free) / vol if vol > 0 else np.nan
    return {
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(value),
    }
