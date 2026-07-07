"""Unit tests for src.backtesting."""

import numpy as np
import pandas as pd
import pytest

from src import backtesting as bt


@pytest.fixture
def flat_returns():
    """Two assets: A gains 1%/day, B loses 1%/day, 10 days."""
    dates = pd.bdate_range("2025-01-01", periods=10)
    return pd.DataFrame({"A": [0.01] * 10, "B": [-0.01] * 10}, index=dates)


def test_buy_and_hold_single_asset(flat_returns):
    value = bt.run_backtest(flat_returns, {"A": 1.0})
    assert np.isclose(value.iloc[-1], 1.01**10)


def test_buy_and_hold_mix_compounds_slices(flat_returns):
    value = bt.run_backtest(flat_returns, {"A": 0.5, "B": 0.5})
    expected = 0.5 * 1.01**10 + 0.5 * 0.99**10
    assert np.isclose(value.iloc[-1], expected)


def test_rebalance_equals_hold_for_single_asset(flat_returns):
    hold = bt.run_backtest(flat_returns, {"A": 1.0})
    reb = bt.run_backtest(flat_returns, {"A": 1.0}, rebalance="ME")
    assert np.allclose(hold.values, reb.values)


def test_missing_asset_raises(flat_returns):
    with pytest.raises(KeyError):
        bt.run_backtest(flat_returns, {"A": 0.5, "MISSING": 0.5})


def test_benchmark_uses_config_weights(flat_returns):
    frame = flat_returns.rename(columns={"A": "SPY", "B": "BND"})
    value = bt.benchmark_portfolio(frame)
    expected = 0.6 * 1.01**10 + 0.4 * 0.99**10
    assert np.isclose(value.iloc[-1], expected)


def test_max_drawdown_monotonic_up_is_zero():
    value = pd.Series(np.linspace(1.0, 2.0, 50))
    assert bt.max_drawdown(value) == 0.0


def test_max_drawdown_known_drop():
    value = pd.Series([1.0, 1.5, 0.75, 1.2])
    assert np.isclose(bt.max_drawdown(value), 0.75 / 1.5 - 1)  # -50%


def test_performance_metrics_keys_and_signs(flat_returns):
    value = bt.run_backtest(flat_returns, {"A": 1.0})
    m = bt.performance_metrics(value)
    assert set(m) == {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    }
    assert m["total_return"] > 0
    assert m["max_drawdown"] == 0.0
    assert np.isclose(m["total_return"], 1.01**9 - 1)  # 9 return observations
