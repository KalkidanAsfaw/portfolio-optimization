"""Task 1: fetch and cache historical financial data from YFinance."""

from __future__ import annotations

import pandas as pd

from . import config


def fetch_data(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    """Download OHLCV + Adj Close for the given tickers from YFinance.

    Returns a long/tidy or wide DataFrame indexed by Date. Implementation to
    be filled in Task 1.
    """
    raise NotImplementedError


def load_cached(path: str | None = None) -> pd.DataFrame:
    """Load previously saved data from data/raw or data/processed."""
    raise NotImplementedError


def save_raw(df: pd.DataFrame, name: str = "prices.csv") -> None:
    """Persist a downloaded DataFrame to data/raw."""
    raise NotImplementedError
