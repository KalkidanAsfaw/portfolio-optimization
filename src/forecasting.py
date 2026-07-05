"""Task 2 & 3: ARIMA/SARIMA and LSTM forecasting models for TSLA prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --- Data prep ---------------------------------------------------------------
def load_target(
    ticker: str = config.FORECAST_TICKER,
    field: str = "Adj Close",
) -> pd.Series:
    """Load the cleaned price series for the forecast target from processed data."""
    from . import preprocessing as pp

    wide = pd.read_csv(
        config.PROCESSED_DIR / "adj_close_wide.csv", index_col=0, parse_dates=True
    )
    if ticker in wide.columns:
        return wide[ticker].dropna()
    # Fallback: rebuild from cached tidy data.
    from . import data_loader as dl

    tidy = pp.clean(dl.get_prices())
    return pp.to_wide(tidy, field)[ticker].dropna()


def chronological_split(
    series: pd.Series, train_end: str = config.TRAIN_END
) -> tuple[pd.Series, pd.Series]:
    """Split a series into train/test at ``train_end``, preserving order.

    Train = up to and including ``train_end``; test = everything after.
    No shuffling — temporal order is preserved (critical for time series).
    """
    train = series.loc[:train_end]
    test = series.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):]
    return train, test


# --- Evaluation --------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and MAPE for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


# --- ARIMA / SARIMA ----------------------------------------------------------
def fit_auto_arima(train: pd.Series, seasonal: bool = False, m: int = 5, **kwargs):
    """Fit an ARIMA/SARIMA model via ``pmdarima.auto_arima`` grid search.

    Operates on the price level; ``d`` (differencing) is chosen automatically
    to handle the non-stationarity found in Task 1.
    """
    import pmdarima as pm

    defaults = dict(
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    defaults.update(kwargs)
    return pm.auto_arima(train.values, **defaults)


def arima_forecast(model, n_periods: int, index: pd.Index | None = None):
    """Forecast ``n_periods`` ahead; return ``(forecast, conf_int)`` frames."""
    fc, conf = model.predict(n_periods=n_periods, return_conf_int=True)
    fc = pd.Series(fc, index=index, name="forecast")
    conf = pd.DataFrame(conf, index=index, columns=["lower", "upper"])
    return fc, conf


# --- LSTM --------------------------------------------------------------------
def make_sequences(values: np.ndarray, window: int = config.LSTM_WINDOW):
    """Build (X, y) sliding-window sequences: ``window`` days -> next day."""
    values = np.asarray(values, dtype=float).reshape(-1)
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window : i])
        y.append(values[i])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def build_lstm(window: int = config.LSTM_WINDOW, units: int = 50, layers: int = 2):
    """Construct the LSTM: input -> LSTM layer(s) -> Dense(1)."""
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf_set_seed()
    model = Sequential()
    model.add(Input(shape=(window, 1)))
    for i in range(layers):
        return_seq = i < layers - 1
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def tf_set_seed(seed: int = config.RANDOM_SEED):
    """Seed numpy + tensorflow for reproducible LSTM runs."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_lstm_pipeline(
    train: pd.Series,
    test: pd.Series,
    window: int = config.LSTM_WINDOW,
    units: int = 50,
    layers: int = 2,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 0,
):
    """End-to-end LSTM: scale, train, one-step-ahead predict over the test set.

    Test predictions use the true trailing ``window`` of actual prices for each
    step (standard walk-forward evaluation), then inverse-transform to price.
    Returns ``(pred_series, fitted_model, history, scaler)``.
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).reshape(-1)

    X_train, y_train = make_sequences(train_scaled, window)
    model = build_lstm(window, units=units, layers=layers)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=verbose,
    )

    # Build test sequences from the last `window` of train + the test series.
    combined = np.concatenate([train.values[-window:], test.values])
    combined_scaled = scaler.transform(combined.reshape(-1, 1)).reshape(-1)
    X_test, _ = make_sequences(combined_scaled, window)

    pred_scaled = model.predict(X_test, verbose=verbose)
    pred = scaler.inverse_transform(pred_scaled).reshape(-1)
    pred_series = pd.Series(pred, index=test.index, name="lstm_pred")
    return pred_series, model, history, scaler


# --- Future forecasting (Task 3) --------------------------------------------
def lstm_forecast_future(model, scaler, history_series: pd.Series, horizon: int,
                         window: int = config.LSTM_WINDOW) -> pd.Series:
    """Iteratively forecast ``horizon`` business days ahead by feeding predictions back."""
    seq = scaler.transform(history_series.values[-window:].reshape(-1, 1)).reshape(-1)
    preds = []
    for _ in range(horizon):
        x = seq[-window:].reshape(1, window, 1)
        p = float(model.predict(x, verbose=0).reshape(-1)[0])
        preds.append(p)
        seq = np.append(seq, p)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).reshape(-1)
    idx = pd.bdate_range(history_series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(preds, index=idx, name="lstm_future")
