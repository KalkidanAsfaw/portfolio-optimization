"""Task 1: cleaning, feature engineering, and scaling."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config


def to_wide(tidy: pd.DataFrame, field: str = "Adj Close") -> pd.DataFrame:
    """Pivot tidy data to a wide frame: Date index, one column per ticker."""
    wide = tidy.pivot(index="Date", columns="Ticker", values=field)
    wide = wide.sort_index()
    return wide[config.TICKERS] if set(config.TICKERS).issubset(wide.columns) else wide


def clean(tidy: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes, sort, and handle missing values on tidy price data.

    Missing values are forward-filled within each ticker (a market holiday /
    gap carries the last known price), then any leading NaNs are dropped.
    """
    df = tidy.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Ticker", "Date"])
    df[num_cols] = df.groupby("Ticker")[num_cols].ffill()
    df = df.dropna(subset=["Adj Close"]).reset_index(drop=True)
    return df


def daily_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns from a wide price frame."""
    return wide_prices.pct_change().dropna(how="all")


def log_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns (useful for stationarity / modeling)."""
    import numpy as np

    return np.log(wide_prices / wide_prices.shift(1)).dropna(how="all")


def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling standard deviation of returns (default ~1 trading month)."""
    return returns.rolling(window=window).std()


def rolling_mean(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling mean of prices."""
    return prices.rolling(window=window).mean()


def scale(df: pd.DataFrame, method: str = "minmax"):
    """Scale a numeric frame. Returns ``(scaler, scaled_df)``.

    ``minmax`` for neural nets (LSTM), ``standard`` for z-scoring.
    """
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    scaled = scaler.fit_transform(df.values)
    return scaler, pd.DataFrame(scaled, index=df.index, columns=df.columns)
