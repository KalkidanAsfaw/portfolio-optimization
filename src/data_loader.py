"""Task 1: fetch and cache historical financial data from YFinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config


def fetch_data(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download OHLCV + Adj Close for the given tickers from YFinance.

    Returns a tidy (long) DataFrame with columns:
    ``Date, Ticker, Open, High, Low, Close, Adj Close, Volume``.

    ``auto_adjust=False`` keeps a distinct ``Adj Close`` column, as required
    by the challenge data spec.
    """
    tickers = tickers or config.TICKERS
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("YFinance returned no data; check tickers/date range.")

    frames = []
    for ticker in tickers:
        # With multiple tickers columns are a MultiIndex (ticker, field).
        sub = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else raw.copy()
        sub = sub.reset_index()
        sub.insert(1, "Ticker", ticker)
        frames.append(sub)

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return tidy


def save_raw(df: pd.DataFrame, name: str = "prices.csv") -> Path:
    """Persist a downloaded DataFrame to ``data/raw``."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / name
    df.to_csv(path, index=False)
    return path


def load_cached(name: str = "prices.csv", processed: bool = False) -> pd.DataFrame:
    """Load previously saved data from ``data/raw`` (or ``data/processed``)."""
    base = config.PROCESSED_DIR if processed else config.RAW_DIR
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"No cached file at {path}. Run fetch_data + save_raw first.")
    return pd.read_csv(path, parse_dates=["Date"])


def get_prices(name: str = "prices.csv", refresh: bool = False) -> pd.DataFrame:
    """Return tidy price data, fetching from YFinance only if not cached."""
    path = config.RAW_DIR / name
    if refresh or not path.exists():
        df = fetch_data()
        save_raw(df, name)
        return df
    return load_cached(name)
