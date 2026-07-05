"""Unit tests for src.preprocessing."""

import numpy as np
import pandas as pd

from src import preprocessing as pp


def test_clean_sorts_and_fills(tidy_prices):
    cleaned = pp.clean(tidy_prices)
    # No missing Adj Close after forward-fill.
    assert cleaned["Adj Close"].isna().sum() == 0
    # Sorted by ticker then date.
    for _, grp in cleaned.groupby("Ticker"):
        assert grp["Date"].is_monotonic_increasing
    # Date column is datetime.
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])


def test_to_wide_shape(tidy_prices):
    cleaned = pp.clean(tidy_prices)
    wide = pp.to_wide(cleaned, "Adj Close")
    assert set(wide.columns) == {"AAA", "BBB"}
    assert wide.index.name == "Date"
    assert wide.isna().sum().sum() == 0


def test_daily_returns_constant_prices():
    prices = pd.DataFrame({"X": [10.0, 10.0, 10.0]})
    rets = pp.daily_returns(prices)
    assert (rets["X"] == 0).all()


def test_daily_returns_values(wide_prices):
    rets = pp.daily_returns(wide_prices)
    # First return of AAA: 101/100 - 1 = 0.01
    assert np.isclose(rets["AAA"].iloc[0], 0.01)
    assert len(rets) == len(wide_prices) - 1


def test_log_returns_match_pct_for_small(wide_prices):
    log_r = pp.log_returns(wide_prices)
    assert log_r.shape[0] == len(wide_prices) - 1
    # log return of increasing series is positive
    assert (log_r["AAA"] > 0).all()


def test_rolling_volatility_window(returns_frame):
    vol = pp.rolling_volatility(returns_frame, window=21)
    # First 20 rows are NaN (need full window).
    assert vol["AAA"].iloc[:20].isna().all()
    assert not np.isnan(vol["AAA"].iloc[21])


def test_scale_minmax_range(wide_prices):
    scaler, scaled = pp.scale(wide_prices, method="minmax")
    assert scaled.shape == wide_prices.shape
    assert scaled.min().min() >= -1e-9
    assert scaled.max().max() <= 1 + 1e-9


def test_scale_standard_mean_zero(wide_prices):
    _, scaled = pp.scale(wide_prices, method="standard")
    assert np.allclose(scaled.mean().values, 0, atol=1e-9)
