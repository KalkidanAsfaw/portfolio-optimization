"""Task 4: Modern Portfolio Theory / efficient frontier via scipy.optimize."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config


def expected_returns(
    returns: pd.DataFrame,
    tsla_view: float,
    periods: int = config.TRADING_DAYS,
) -> pd.Series:
    """Expected-return vector: forecast view for TSLA, historical for the rest.

    ``tsla_view`` is the annualized expected return from the Task 3 forecast;
    BND/SPY use their historical mean daily return, annualized.
    """
    mu = returns.mean() * periods
    mu.loc[config.FORECAST_TICKER] = tsla_view
    return mu


def covariance_matrix(
    returns: pd.DataFrame, periods: int = config.TRADING_DAYS
) -> pd.DataFrame:
    """Annualized covariance matrix of daily returns."""
    return returns.cov() * periods


def portfolio_performance(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = config.RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """Return (expected return, volatility, Sharpe ratio) for given weights."""
    w = np.asarray(weights)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _optimize(objective, n_assets: int) -> np.ndarray:
    """Long-only, fully-invested weight optimization."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float = config.RISK_FREE_RATE
) -> pd.Series:
    """Tangency (maximum Sharpe ratio) portfolio weights."""
    def neg_sharpe(w):
        ret, vol, _ = portfolio_performance(w, mu, cov, risk_free)
        return -(ret - risk_free) / vol

    w = _optimize(neg_sharpe, len(mu))
    return pd.Series(w, index=mu.index, name="max_sharpe")


def min_volatility_portfolio(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Minimum-volatility portfolio weights."""
    def vol(w):
        return np.sqrt(np.asarray(w) @ cov.values @ np.asarray(w))

    w = _optimize(vol, len(mu))
    return pd.Series(w, index=mu.index, name="min_volatility")


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 60
) -> pd.DataFrame:
    """Trace the efficient frontier: min volatility for each target return.

    Returns a frame with columns ``return``, ``volatility`` and one weight
    column per asset.
    """
    n = len(mu)
    min_ret = float(min_volatility_portfolio(mu, cov) @ mu)
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(np.asarray(w) @ mu.values) - t},
        ]
        result = minimize(
            lambda w: np.sqrt(np.asarray(w) @ cov.values @ np.asarray(w)),
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=constraints,
        )
        if result.success:
            ret, vol, _ = portfolio_performance(result.x, mu, cov)
            rows.append({"return": ret, "volatility": vol, **dict(zip(mu.index, result.x))})
    return pd.DataFrame(rows)


def random_portfolios(
    mu: pd.Series, cov: pd.DataFrame, n: int = 5000, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Monte-Carlo cloud of random long-only portfolios (for frontier context)."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(mu)), size=n)
    rets = weights @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov.values, weights))
    sharpes = (rets - config.RISK_FREE_RATE) / vols
    return pd.DataFrame({"return": rets, "volatility": vols, "sharpe": sharpes})
