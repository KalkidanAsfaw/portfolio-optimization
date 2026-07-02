"""Task 1: exploratory analysis, stationarity tests, and risk metrics."""

from __future__ import annotations

import pandas as pd


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test. Returns stat, p-value, etc."""
    raise NotImplementedError


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (VaR)."""
    raise NotImplementedError


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Annualized Sharpe ratio from a return series."""
    raise NotImplementedError


def detect_outliers(returns: pd.Series, z: float = 3.0) -> pd.Series:
    """Flag unusually high/low return days."""
    raise NotImplementedError
