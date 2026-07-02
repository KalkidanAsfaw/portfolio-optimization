# Time Series Forecasting for Portfolio Management Optimization

**GMF Investments — 10 Academy KAIM Week 9**

Apply time series forecasting to historical financial data (TSLA, BND, SPY) to
enhance portfolio management: forecast trends, optimize asset allocation with
Modern Portfolio Theory, and validate the strategy through backtesting.

## Assets

| Asset | Ticker | Role | Risk Profile |
| :--- | :--- | :--- | :--- |
| Tesla | TSLA | High-growth stock (forecasted asset) | High risk / high return |
| Vanguard Total Bond Market ETF | BND | U.S. investment-grade bonds | Low risk / stability |
| S&P 500 ETF | SPY | Broad market exposure | Moderate risk |

Data: YFinance, **2015-01-01 → 2026-06-30**.

## Project Structure

```
portfolio-optimization/
├── .github/workflows/unittests.yml   # CI
├── .vscode/settings.json
├── data/
│   ├── raw/                          # downloaded YFinance data (gitignored)
│   └── processed/                    # cleaned/feature-engineered data
├── notebooks/                        # one notebook per task (01–05)
├── src/                              # reusable package
│   ├── config.py                     # paths, tickers, dates, constants
│   ├── data_loader.py                # Task 1: fetch/cache
│   ├── preprocessing.py              # Task 1: clean, returns, scaling
│   ├── eda.py                        # Task 1: stationarity, VaR, Sharpe
│   ├── forecasting.py                # Task 2/3: ARIMA/SARIMA, LSTM
│   ├── optimization.py               # Task 4: MPT / efficient frontier
│   └── backtesting.py                # Task 5: strategy vs. benchmark
├── scripts/                         # CLI entry points / pipeline runners
├── tests/                           # pytest unit tests
├── requirements.txt
└── PLAN.md                          # detailed task plan & timeline
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tasks

1. **Preprocess & Explore** — fetch, clean, EDA, stationarity (ADF), risk metrics (VaR, Sharpe).
2. **Forecasting Models** — ARIMA/SARIMA vs. LSTM; chronological split; MAE/RMSE/MAPE.
3. **Forecast Future Trends** — 6–12 month forecast with confidence intervals + trend analysis.
4. **Portfolio Optimization** — efficient frontier, max-Sharpe & min-volatility portfolios.
5. **Backtesting** — strategy vs. 60/40 SPY/BND benchmark.

See [PLAN.md](PLAN.md) for the full breakdown and timeline.

## Testing

```bash
pytest tests -v
```
