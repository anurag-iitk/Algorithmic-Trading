# Assignment 5 — Historical Intraday Data Store for the Top 50 S&P 500 Stocks

Python program that uses the **Yahoo Finance REST API** (via `yfinance`) to fetch **intraday
5-minute price-volume data** for the **top 50 stocks of the S&P 500**, and **incrementally
builds a persistent historical data store** — one CSV per stock — handling missing data and
storing an outlier-treated copy alongside.

## What it does

1. **Fetch top-50 list live** — ranked by index weight from slickcharts.com (Wikipedia and a
   built-in list as fallbacks).
2. **Download** — intraday (default 5-minute) OHLCV bars for each stock from Yahoo Finance,
   in date chunks.
3. **Build a historical store (the key idea)** — for each stock it loads the existing CSV,
   **merges** the freshly fetched bars into it (de-duplicating timestamps), and writes it
   back. The **first run backfills** the maximum available history; **later runs only fetch a
   small recent window** and append new bars. Run it periodically (e.g. via cron) and the
   store grows over time.
4. **Clean / handle missing data** — removes duplicate timestamps, fills missing values (time
   interpolation + forward/backward fill), drops invalid rows, derives **adjusted** prices.
5. **Store** — each stock in its **own CSV** under `data/cleaned/`, plus a **separate
   outlier-treated copy** under `data/outlier_treated/`.
6. **Audit** — `data/summary.csv` records, per stock, how many **new bars** each run added.

## ⚠️ Yahoo Finance 5-minute data is limited to ~60 days

Yahoo serves 5-minute data for only ~60 days, so a single run backfills at most ~60 days.
**Run the script regularly to accumulate a longer history than Yahoo alone exposes** — each
run appends the latest bars to the store. To backfill a genuine multi-year intraday store in
one shot, point the same pipeline at a broker REST API (e.g. **Alpaca**, **Polygon.io**) and
raise `--max-intraday-days`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Update the store for the top 50 stocks (first run backfills ~60 days)
python build_historical_store.py

# Quick subset for testing
python build_historical_store.py --top 3

# Re-backfill from scratch, ignoring the existing store
python build_historical_store.py --full-refresh

# Explicit tickers
python build_historical_store.py --tickers AAPL,MSFT,NVDA
```

Useful options: `--interval`, `--backfill-days`, `--overlap-days`, `--max-intraday-days`,
`--chunk-days`, `--full-refresh`, `--top`, `--source`, `--tickers`, `--tickers-file`,
`--output-dir`, `--threshold`. Run `python build_historical_store.py --help` for all options.

## Output layout

```
data/
├── cleaned/            # historical store: one CSV per stock (e.g. AAPL.csv)
├── outlier_treated/    # outlier-treated copy per stock (adds an Outlier_Treated flag)
└── summary.csv         # per-stock audit: rows_before, rows_added, rows_total, outliers, range
```

Each CSV contains: `Datetime, Open, High, Low, Close, Volume, Adj Open, Adj High, Adj Low,
Adj Close` (the outlier-treated files add an `Outlier_Treated` column). `Datetime` is
timezone-aware (`America/New_York`) and covers regular trading hours only.

## How the incremental store works

- **First run / `--full-refresh`** → fetches `--backfill-days` (default ~59) of history.
- **Later runs** → fetches only from `last_stored_timestamp - --overlap-days` to now; the
  small overlap re-checks recent bars so any revised values are updated, then everything is
  merged and **de-duplicated by timestamp** (newer bars win). This makes re-runs **idempotent**
  and cheap.

## Notes

- The top-50 list is fetched **live** (weight-ranked); override with `--source` or `--tickers`.
- Outliers are treated on **intraday log-returns** with the **overnight gap excluded** (an
  overnight jump is a real gap, not a bad tick); the store keeps the un-treated data intact.
