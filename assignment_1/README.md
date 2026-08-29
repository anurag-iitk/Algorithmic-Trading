# Assignment 1 — Historical Price-Volume Data of Top 50 S&P 500 Stocks

Python program that downloads, cleans and stores **10 years of daily (EOD) price-volume
data** for the **top 50 stocks of the S&P 500** index from **Yahoo Finance**.

## What it does

1. **Download** — last 10 years of daily OHLCV data for each stock via the `yfinance` API.
2. **Clean / process** — for every stock:
   - removes **duplicate** trading dates,
   - handles **missing data** (time interpolation + forward/backward fill),
   - removes **invalid rows** (non-positive prices, `High < Low`),
   - derives fully **adjusted** OHLC prices from the split/dividend adjustment factor
     (`Adj Close / Close`).
3. **Store** — each stock is saved to its **own CSV file** in `data/cleaned/`.
4. **Treat outliers** — a **separate copy** of each stock, with extreme daily returns
   winsorised (robust MAD / modified z-score), is saved in `data/outlier_treated/`.
5. **Audit** — a `data/summary.csv` report records what happened to every ticker.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Full assignment: top 50 stocks, 10 years
python fetch_stock_data.py

# Quick test on a small subset
python fetch_stock_data.py --top 5 --years 2

# Explicit tickers
python fetch_stock_data.py --tickers AAPL,MSFT,NVDA
```

Useful options: `--years`, `--top`, `--tickers`, `--tickers-file`, `--output-dir`,
`--threshold` (outlier sensitivity). Run `python fetch_stock_data.py --help` for all options.

## Output layout

```
data/
├── cleaned/            # one cleaned CSV per stock (e.g. AAPL.csv)
├── outlier_treated/    # one outlier-treated CSV per stock (adds an Outlier_Treated flag)
└── summary.csv         # per-ticker audit: rows, duplicates, fills, outliers, date range
```

Each CSV contains: `Date, Open, High, Low, Close, Volume, Adj Open, Adj High, Adj Low, Adj Close`
(the outlier-treated files add an `Outlier_Treated` boolean column).

## Notes

- The S&P 500 is capitalisation-weighted, so the "top 50 stocks" are its 50 largest
  constituents by index weight. This list drifts over time; it is defined in
  `SP500_TOP_50` and can be overridden with `--tickers` / `--tickers-file`.
- Outliers are treated on **daily log-returns** rather than price levels, because prices
  legitimately trend over a 10-year window while returns are approximately stationary.
