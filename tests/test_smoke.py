"""Smoke tests to keep CI green until task logic is implemented."""

import src.config as config


def test_tickers_configured():
    assert config.TICKERS == ["TSLA", "BND", "SPY"]
    assert config.FORECAST_TICKER == "TSLA"


def test_date_range():
    assert config.START_DATE == "2015-01-01"
    assert config.END_DATE == "2026-06-30"


def test_benchmark_weights_sum_to_one():
    assert abs(sum(config.BENCHMARK_WEIGHTS.values()) - 1.0) < 1e-9
