"""Task 2 & 3: ARIMA/SARIMA and LSTM forecasting models."""

from __future__ import annotations

import pandas as pd

from . import config


def chronological_split(series: pd.Series, train_end: str = config.TRAIN_END):
    """Split a series into train/test preserving temporal order."""
    raise NotImplementedError


# --- ARIMA / SARIMA ----------------------------------------------------------
def fit_arima(train: pd.Series):
    """Fit ARIMA/SARIMA (auto_arima grid search) on training data."""
    raise NotImplementedError


# --- LSTM --------------------------------------------------------------------
def make_sequences(series, window: int = config.LSTM_WINDOW):
    """Build (X, y) sliding-window sequences for the LSTM."""
    raise NotImplementedError


def build_lstm(window: int = config.LSTM_WINDOW):
    """Construct the LSTM architecture (input -> LSTM layer(s) -> Dense)."""
    raise NotImplementedError


# --- Evaluation --------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE, MAPE."""
    raise NotImplementedError


def forecast_future(model, horizon: int):
    """Multi-step forecast into the future (Task 3)."""
    raise NotImplementedError
