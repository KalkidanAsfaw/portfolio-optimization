"""Task 1: cleaning, feature engineering, and scaling."""

from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values/dates, enforce dtypes, sort by date."""
    raise NotImplementedError


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    raise NotImplementedError


def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling standard deviation of returns."""
    raise NotImplementedError


def scale(df: pd.DataFrame):
    """Normalize/standardize features (returns scaler + scaled data)."""
    raise NotImplementedError
