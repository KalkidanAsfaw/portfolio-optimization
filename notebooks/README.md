# Notebooks

Analysis notebooks, one per challenge task. Notebooks import reusable logic
from the `src/` package rather than duplicating it.

| Notebook | Task | Purpose |
| :--- | :--- | :--- |
| `01_eda_preprocessing.ipynb` | Task 1 | Fetch, clean, EDA, stationarity, risk metrics |
| `02_forecasting_models.ipynb` | Task 2 | ARIMA/SARIMA & LSTM, train/test, metrics |
| `03_future_forecast.ipynb` | Task 3 | 6–12 month forecast with confidence intervals |
| `04_portfolio_optimization.ipynb` | Task 4 | Efficient frontier, key portfolios |
| `05_backtesting.ipynb` | Task 5 | Strategy vs. 60/40 benchmark |

## Setup

From the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name portfolio-opt
jupyter notebook
```
