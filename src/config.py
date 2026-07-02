"""Central configuration shared across tasks."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Assets ------------------------------------------------------------------
TICKERS = ["TSLA", "BND", "SPY"]
FORECAST_TICKER = "TSLA"  # asset we forecast (Task 2/3)

# --- Date range (challenge spec) --------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

# --- Chronological split -----------------------------------------------------
TRAIN_END = "2024-12-31"   # train: 2015-2024
TEST_START = "2025-01-01"  # test:  2025-2026

# --- Finance constants -------------------------------------------------------
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02      # annual, for Sharpe ratio
VAR_CONFIDENCE = 0.95

# --- Modeling ----------------------------------------------------------------
LSTM_WINDOW = 60           # lookback days -> predict next day
RANDOM_SEED = 42

# --- Backtest ----------------------------------------------------------------
BACKTEST_START = "2025-01-01"
BENCHMARK_WEIGHTS = {"SPY": 0.60, "BND": 0.40}
