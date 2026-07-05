"""Unit tests for src.eda (stationarity + risk metrics)."""

import numpy as np
import pandas as pd

from src import eda


def test_adf_stationary_white_noise():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500))
    res = eda.adf_test(s, name="wn")
    assert res["stationary"] is True
    assert res["p_value"] < 0.05
    assert isinstance(res["stationary"], bool)
    assert set(["adf_statistic", "p_value", "critical_values"]).issubset(res)


def test_adf_nonstationary_trend():
    # Strong linear trend plus small noise (noise avoids a zero-residual fit).
    rng = np.random.default_rng(7)
    s = pd.Series(np.arange(500.0) + rng.normal(0, 1, 500))
    res = eda.adf_test(s)
    assert res["stationary"] is False
    assert res["p_value"] > 0.05


def test_adf_report_columns(returns_frame):
    rep = eda.adf_report(returns_frame)
    assert list(rep.index) == ["AAA", "BBB"]
    assert "p_value" in rep.columns and "stationary" in rep.columns


def test_value_at_risk_matches_percentile():
    rets = pd.Series(np.linspace(-0.1, 0.1, 101))
    var95 = eda.value_at_risk(rets, confidence=0.95)
    assert np.isclose(var95, np.percentile(rets, 5))
    assert var95 < 0


def test_sharpe_positive_for_positive_mean():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.002, 0.01, 500))
    assert eda.sharpe_ratio(rets, risk_free=0.0) > 0


def test_sharpe_nan_for_zero_variance():
    rets = pd.Series([0.001] * 50)
    assert np.isnan(eda.sharpe_ratio(rets))


def test_annualized_return():
    rets = pd.Series([0.001] * 252)
    assert np.isclose(eda.annualized_return(rets, periods=252), 0.252)


def test_detect_outliers_flags_spike(returns_frame):
    out = eda.detect_outliers(returns_frame["AAA"], z=3.0)
    assert len(out) >= 1
    # The injected 0.5 spike (row 100) must be flagged.
    assert returns_frame.index[100] in out.index
    assert {"return", "zscore"}.issubset(out.columns)


def test_risk_summary_structure(returns_frame):
    summary = eda.risk_summary(returns_frame)
    assert list(summary.index) == ["AAA", "BBB"]
    assert set(summary.columns) == {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "var_95",
    }
