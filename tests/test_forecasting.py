"""Unit tests for src.forecasting (split, metrics, sequences, LSTM shape)."""

import numpy as np
import pandas as pd
import pytest

from src import forecasting as fc


def _series(n=400, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(np.linspace(100, 200, n), index=idx)


def test_chronological_split_no_overlap():
    s = _series(n=3000)  # spans past 2024-12-31
    train, test = fc.chronological_split(s, train_end="2024-12-31")
    assert train.index.max() <= pd.Timestamp("2024-12-31")
    assert test.index.min() > pd.Timestamp("2024-12-31")
    # Disjoint and complete.
    assert len(train.index.intersection(test.index)) == 0
    assert len(train) + len(test) == len(s)


def test_chronological_split_preserves_order():
    s = _series(n=3000)
    train, test = fc.chronological_split(s, train_end="2024-12-31")
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_evaluate_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = fc.evaluate(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["MAPE"] == 0


def test_evaluate_known_values():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([11.0, 9.0])
    m = fc.evaluate(y_true, y_pred)
    assert np.isclose(m["MAE"], 1.0)
    assert np.isclose(m["RMSE"], 1.0)
    assert np.isclose(m["MAPE"], 10.0)


def test_make_sequences_shapes():
    values = np.arange(10.0)
    X, y = fc.make_sequences(values, window=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    # First window is [0,1,2] predicting 3.
    assert np.array_equal(X[0].reshape(-1), np.array([0, 1, 2]))
    assert y[0] == 3


def test_make_sequences_last_target():
    values = np.arange(10.0)
    _, y = fc.make_sequences(values, window=3)
    assert y[-1] == 9


def test_build_lstm_output_shape():
    tf = pytest.importorskip("tensorflow")
    model = fc.build_lstm(window=60, units=8, layers=2)
    assert model.input_shape == (None, 60, 1)
    assert model.output_shape == (None, 1)
