# Assignment 2 — Intraday (5-Minute) Price-Volume Data of Top 50 S&P 500 Stocks

Python program that downloads, cleans and stores **intraday 5-minute price-volume data**
for the **top 50 stocks of the S&P 500** index from **Yahoo Finance**.

## What it does

1. **Download** — intraday (default 5-minute) OHLCV bars for each stock via the `yfinance`
   API, fetched in date **chunks** and stitched together. The top-50 constituent list is
   fetched **live** (ranked by index weight from slickcharts.com, with Wikipedia and a
   built-in list as fallbacks).
2. **Clean / process** — for every stock:
   - removes **duplicate** timestamps,
   - handles **missing data** (time interpolation + forward/backward fill),
   - removes **invalid rows** (non-positive prices, `High < Low`),
   - derives fully **adjusted** OHLC prices from the split/dividend adjustment factor
     (`Adj Close / Close`).
3. **Store** — each stock is saved to its **own CSV file** in `data/cleaned/`.
4. **Treat outliers** — a **separate copy** of each stock, with extreme **intraday** returns
   winsorised (robust MAD / modified z-score, overnight gaps excluded), is saved in
   `data/outlier_treated/`.
5. **Audit** — a `data/summary.csv` report records what happened to every ticker.

## ⚠️ Important: Yahoo Finance 5-minute data is limited to ~60 days

The assignment asks for **1 year** of 5-minute data. **Yahoo Finance only provides
5-minute data for roughly the last 60 days** — any longer request is rejected with
*"5m data not available … must be within the last 60 days"*.

This program handles that honestly:

- it **requests 1 year** (`--days 365`) but automatically **caps** the intraday window to
  Yahoo's limit (`--max-intraday-days`, default 59) and logs a clear warning;
- the download is written as **chunked date-range requests**, which is exactly the technique
  needed to page through long intraday history on a broker REST API.

To obtain a **genuine full year** of 5-minute data, point this same pipeline at a broker
REST API that serves longer intraday history (e.g. **Alpaca**, **Polygon.io**, **Tiingo**)
and raise `--max-intraday-days` — no other code changes are required.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Full assignment: top 50 stocks, 5-minute bars (capped to Yahoo's ~60-day window)
python fetch_intraday_data.py

# Quick test on a small subset
python fetch_intraday_data.py --top 3

# Explicit tickers / a different intraday resolution
python fetch_intraday_data.py --tickers AAPL,MSFT,NVDA
python fetch_intraday_data.py --interval 15m

# Choose where the S&P 500 list comes from
python fetch_intraday_data.py --source slickcharts   # live, weight-ranked (default via auto)
python fetch_intraday_data.py --source static        # built-in list, fully offline
```

Useful options: `--days`, `--interval`, `--max-intraday-days`, `--chunk-days`, `--top`,
`--source`, `--tickers`, `--tickers-file`, `--output-dir`, `--threshold` (outlier sensitivity).
Run `python fetch_intraday_data.py --help` for all options.

## Output layout

```
data/
├── cleaned/            # one cleaned CSV per stock (e.g. AAPL.csv)
├── outlier_treated/    # one outlier-treated CSV per stock (adds an Outlier_Treated flag)
└── summary.csv         # per-ticker audit: rows, days, duplicates, fills, outliers, range
```

Each CSV contains: `Datetime, Open, High, Low, Close, Volume, Adj Open, Adj High, Adj Low,
Adj Close` (the outlier-treated files add an `Outlier_Treated` boolean column). `Datetime`
is timezone-aware (`America/New_York`, the exchange timezone) and covers regular trading
hours only (09:30–16:00 ET).

## Notes

- The S&P 500 is capitalisation-weighted, so the "top 50 stocks" are its 50 largest
  constituents by index weight. The list is fetched **live** by default (weight-ranked from
  slickcharts.com); `--source wikipedia` uses Wikipedia's membership list (unranked) and
  `--source static` uses the built-in `SP500_TOP_50` fallback. Any network failure falls
  back automatically so the pipeline still runs offline. Override entirely with `--tickers`
  / `--tickers-file`.
- Outliers are treated on **intraday log-returns**, and the **overnight gap** (the first bar
  of each session) is deliberately excluded — an overnight jump is a real price gap, not a
  bad tick, so it must not be flagged or altered.
