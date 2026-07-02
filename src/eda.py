"""Task 1: exploratory analysis, stationarity tests, and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from . import config


def adf_test(series: pd.Series, name: str | None = None) -> dict:
    """Augmented Dickey-Fuller stationarity test.

    Null hypothesis: the series has a unit root (non-stationary). A p-value
    below 0.05 lets us reject it and treat the series as stationary.
    """
    series = series.dropna()
    stat, pvalue, nlags, nobs, crit, _ = adfuller(series, autolag="AIC")
    return {
        "series": name,
        "adf_statistic": stat,
        "p_value": pvalue,
        "n_lags": nlags,
        "n_obs": nobs,
        "critical_values": crit,
        "stationary": pvalue < 0.05,
    }


def adf_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the ADF test on every column and return a tidy summary table."""
    rows = [adf_test(frame[col], name=col) for col in frame.columns]
    out = pd.DataFrame(rows).drop(columns=["critical_values"])
    return out.set_index("series")


def value_at_risk(returns: pd.Series, confidence: float = config.VAR_CONFIDENCE) -> float:
    """Historical Value at Risk: the loss threshold at the given confidence.

    Returns a negative number (e.g. -0.05 = 5% one-day loss not exceeded with
    ``confidence`` probability).
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
    periods: int = config.TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio from a daily return series."""
    r = returns.dropna()
    excess = r.mean() * periods - risk_free
    vol = r.std() * np.sqrt(periods)
    return float(excess / vol) if vol else np.nan


def annualized_return(returns: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Annualized mean return."""
    return float(returns.dropna().mean() * periods)


def annualized_volatility(returns: pd.Series, periods: int = config.TRADING_DAYS) -> float:
    """Annualized volatility (std)."""
    return float(returns.dropna().std() * np.sqrt(periods))


def detect_outliers(returns: pd.Series, z: float = 3.0) -> pd.DataFrame:
    """Flag unusually high/low return days via z-score.

    Returns a frame of outlier dates with their return and z-score.
    """
    r = returns.dropna()
    zscores = (r - r.mean()) / r.std()
    mask = zscores.abs() > z
    return pd.DataFrame({"return": r[mask], "zscore": zscores[mask]}).sort_values("zscore")


def risk_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset table of annualized return, volatility, Sharpe, and 95% VaR."""
    rows = {}
    for col in returns.columns:
        s = returns[col]
        rows[col] = {
            "annual_return": annualized_return(s),
            "annual_volatility": annualized_volatility(s),
            "sharpe": sharpe_ratio(s),
            "var_95": value_at_risk(s),
        }
    return pd.DataFrame(rows).T
