# Project Plan — Portfolio Management Optimization

**Window:** 01 Jul → 07 Jul 2026
**Interim (Task 1 + one model):** Sun 05 Jul, 20:00 UTC
**Final (all tasks + memo):** Tue 07 Jul, 20:00 UTC

## Approach

Reusable logic lives in `src/` (tested with `pytest`); each notebook (`01`–`05`)
imports from `src/` and produces the deliverable visualizations. Data cached in
`data/raw` so YFinance is hit once. TSLA is the forecasted asset; BND/SPY use
historical returns.

---

## Task 1 — Preprocess & Explore  → `01_eda_preprocessing.ipynb`, `data_loader.py`, `preprocessing.py`, `eda.py`

- [ ] Fetch TSLA/BND/SPY via YFinance (2015-01-01 → 2026-06-30); cache to `data/raw`.
- [ ] Clean: dtypes, missing dates/values (interpolate/ffill), sort by date.
- [ ] EDA: closing-price trends, daily % change, rolling mean/std volatility.
- [ ] Outlier detection on returns (z-score / extreme days).
- [ ] Stationarity: ADF test on close prices **and** daily returns; interpret (→ `d`).
- [ ] Risk metrics: Value at Risk (95%), annualized Sharpe ratio.
- **Deliverables:** notebook, data-quality summary, ADF results, ≥3 visualizations.

## Task 2 — Forecasting Models  → `02_forecasting_models.ipynb`, `forecasting.py`

- [ ] Chronological split: train 2015–2024, test 2025–2026 (no shuffling).
- [ ] ARIMA/SARIMA via `auto_arima` (or ACF/PACF); document (p,d,q)(P,D,Q,m).
- [ ] LSTM: 60-day windows, scaling, LSTM layer(s) → Dense; tune epochs/batch/lr.
- [ ] Evaluate both on test set: MAE, RMSE, MAPE → comparison table.
- [ ] Discuss which model wins and why.
- **Deliverables:** trained ARIMA + LSTM, metrics table, selection rationale.
- **Interim target:** at least the ARIMA model complete by Sun 05 Jul.

## Task 3 — Forecast Future Trends  → `03_future_forecast.ipynb`

- [ ] Best model → 6–12 month future forecast.
- [ ] Plot historical vs. test predictions vs. future, with confidence intervals.
- [ ] Trend analysis; discuss how CI width grows over the horizon (reliability).
- [ ] Opportunities & risks from the forecast.
- **Deliverables:** forecast plot w/ CIs, trend summary, opportunities/risks, reliability note.

## Task 4 — Portfolio Optimization (MPT)  → `04_portfolio_optimization.ipynb`, `optimization.py`

- [ ] Expected returns: TSLA from forecast, BND/SPY from historical (annualized).
- [ ] Annualized covariance matrix of daily returns.
- [ ] Efficient frontier via PyPortfolioOpt / scipy.optimize.
- [ ] Mark Max-Sharpe (tangency) & Min-Volatility portfolios.
- [ ] Recommend a portfolio: weights, expected return, volatility, Sharpe + justification.
- **Deliverables:** frontier plot, covariance heatmap, recommendation + 1-paragraph justification.

## Task 5 — Backtesting  → `05_backtesting.ipynb`, `backtesting.py`

- [ ] Backtest window: last year of data (~Jul 2025 → Jun 2026), out-of-sample.
- [ ] Benchmark: static 60% SPY / 40% BND.
- [ ] Simulate strategy (hold initial weights; optional monthly rebalance).
- [ ] Metrics: total return, annualized return, Sharpe, max drawdown.
- [ ] Cumulative returns plot; conclude on viability + limitations.
- **Deliverables:** cumulative-returns plot, metrics table, 1–2 paragraph conclusion.

---

## Cross-cutting

- [ ] `requirements.txt` pinned; venv reproducible.
- [ ] Unit tests in `tests/` for `src/` functions; CI green (`unittests.yml`).
- [ ] README + code docstrings kept current.
- [ ] **Final report:** investment memo (PDF or blog) — methodology per task, model
      comparison, efficient frontier + recommendation, backtest results, screenshots.

## Suggested Timeline

| Day | Focus |
| :--- | :--- |
| Wed–Thu 01–02 Jul | Task 1 (data, EDA, stationarity, risk) |
| Fri 03 Jul | Task 2 ARIMA + start LSTM |
| Sat 04 Jul | Finish Task 2; interim report draft |
| **Sun 05 Jul** | **Interim submission** (Task 1 + a model) |
| Mon 06 Jul | Tasks 3 & 4 |
| Tue 07 Jul | Task 5 + final investment memo → **Final submission** |
