"""Shared fixtures: synthetic price/return data (no network or cached files)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tidy_prices():
    """Tidy OHLCV frame for two tickers with a deliberate gap (NaN) to clean."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    rows = []
    for ticker, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i, d in enumerate(dates):
            price = base + i
            rows.append(
                {
                    "Date": d,
                    "Ticker": ticker,
                    "Open": price,
                    "High": price + 1,
                    "Low": price - 1,
                    "Close": price,
                    "Adj Close": price,
                    "Volume": 1000 + i,
                }
            )
    df = pd.DataFrame(rows)
    # Inject a missing Adj Close to exercise the ffill path.
    df.loc[(df["Ticker"] == "AAA") & (df.index == 3), "Adj Close"] = np.nan
    # Shuffle to verify clean() sorts.
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def wide_prices():
    """Wide price frame, strictly increasing (deterministic returns)."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    return pd.DataFrame(
        {"AAA": np.arange(100, 130.0), "BBB": np.arange(50, 80.0)}, index=dates
    )


@pytest.fixture
def returns_frame():
    """Return frame with a controlled outlier in AAA."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=200)
    data = {
        "AAA": rng.normal(0.001, 0.02, 200),
        "BBB": rng.normal(0.0005, 0.01, 200),
    }
    df = pd.DataFrame(data, index=dates)
    df.iloc[100, 0] = 0.5  # a clear >3-sigma outlier
    return df
