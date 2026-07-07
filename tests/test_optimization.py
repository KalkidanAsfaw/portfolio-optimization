"""Unit tests for src.optimization (MPT / efficient frontier)."""

import numpy as np
import pandas as pd
import pytest

from src import optimization as opt


@pytest.fixture
def three_asset_returns():
    """Synthetic daily returns for TSLA/BND/SPY-like assets."""
    rng = np.random.default_rng(7)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {
            "TSLA": rng.normal(0.002, 0.035, n),
            "BND": rng.normal(0.0001, 0.003, n),
            "SPY": rng.normal(0.0006, 0.011, n),
        },
        index=dates,
    )


def test_expected_returns_uses_view(three_asset_returns):
    mu = opt.expected_returns(three_asset_returns, tsla_view=0.30)
    assert np.isclose(mu["TSLA"], 0.30)
    # Others stay historical (annualized daily mean).
    assert np.isclose(mu["SPY"], three_asset_returns["SPY"].mean() * 252)


def test_covariance_symmetric_annualized(three_asset_returns):
    cov = opt.covariance_matrix(three_asset_returns)
    assert np.allclose(cov.values, cov.values.T)
    assert np.allclose(cov.values, three_asset_returns.cov().values * 252)
    # Positive variance on the diagonal.
    assert (np.diag(cov.values) > 0).all()


def test_portfolio_performance_known_values():
    mu = pd.Series({"A": 0.10, "B": 0.05})
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    ret, vol, sharpe = opt.portfolio_performance(np.array([0.5, 0.5]), mu, cov, risk_free=0.02)
    assert np.isclose(ret, 0.075)
    assert np.isclose(vol, np.sqrt(0.25 * 0.04 + 0.25 * 0.01))
    assert np.isclose(sharpe, (0.075 - 0.02) / vol)


def test_weights_valid(three_asset_returns):
    mu = opt.expected_returns(three_asset_returns, tsla_view=0.20)
    cov = opt.covariance_matrix(three_asset_returns)
    for w in (opt.max_sharpe_portfolio(mu, cov), opt.min_volatility_portfolio(mu, cov)):
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert (w.values >= -1e-9).all() and (w.values <= 1 + 1e-9).all()


def test_min_vol_below_max_sharpe_vol(three_asset_returns):
    mu = opt.expected_returns(three_asset_returns, tsla_view=0.20)
    cov = opt.covariance_matrix(three_asset_returns)
    _, ms_vol, _ = opt.portfolio_performance(opt.max_sharpe_portfolio(mu, cov).values, mu, cov)
    _, mv_vol, _ = opt.portfolio_performance(opt.min_volatility_portfolio(mu, cov).values, mu, cov)
    assert mv_vol <= ms_vol + 1e-9


def test_max_sharpe_beats_random(three_asset_returns):
    mu = opt.expected_returns(three_asset_returns, tsla_view=0.20)
    cov = opt.covariance_matrix(three_asset_returns)
    _, _, best = opt.portfolio_performance(opt.max_sharpe_portfolio(mu, cov).values, mu, cov)
    cloud = opt.random_portfolios(mu, cov, n=2000)
    assert best >= cloud["sharpe"].max() - 1e-6


def test_frontier_monotone_volatility(three_asset_returns):
    mu = opt.expected_returns(three_asset_returns, tsla_view=0.20)
    cov = opt.covariance_matrix(three_asset_returns)
    ef = opt.efficient_frontier(mu, cov, n_points=15)
    assert len(ef) > 5
    # Along the frontier, higher target return never costs less volatility.
    assert (ef.sort_values("return")["volatility"].diff().dropna() >= -1e-6).all()
    # Frontier weight columns are valid portfolios.
    w = ef[mu.index.tolist()].values
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
