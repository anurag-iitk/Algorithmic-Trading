# Assignment 3 — Intraday Price Chart with EMA Technical Indicator

Python program that uses the **Yahoo Finance REST API** (via `yfinance`) to fetch **intraday
5-minute price-volume data** for **a single stock**, **plots the closing prices and the
running n-period EMA**, handles missing data, and stores the data (plus an outlier-treated
copy) as CSV.

## What it does

1. **Fetch** — intraday (default 5-minute) OHLCV bars for one stock (default `AAPL`) from
   Yahoo Finance, downloaded in date chunks.
2. **Clean / handle missing data** — removes duplicate timestamps, fills missing values
   (time interpolation + forward/backward fill), drops invalid rows, and derives **adjusted**
   OHLC prices.
3. **Technical indicator** — computes the running **n-period EMA** of the closing price
   (default **12-period**; supports multiple, e.g. `--ema 12,26`).
4. **Plot** — a price chart of **Close + EMA(s)** with a **volume panel**, saved as a PNG,
   plus a chart comparing prices **before vs after outlier treatment**.
5. **Store** — the cleaned data as CSV, and a **separate outlier-treated copy** as CSV.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Default: AAPL, 5-minute bars, EMA-12, last 5 days
python fetch_plot_indicators.py

# Other examples
python fetch_plot_indicators.py --ticker MSFT --ema 12,26
python fetch_plot_indicators.py --interval 15m --days 30
python fetch_plot_indicators.py --show          # also display the charts interactively
```

Useful options: `--ticker`, `--interval`, `--days`, `--ema`, `--threshold` (outlier
sensitivity), `--output-dir`, `--charts-dir`, `--show`. Run
`python fetch_plot_indicators.py --help` for all options.

## Output layout

```
data/
├── <TICKER>_cleaned.csv           # cleaned OHLCV + adjusted prices + EMA column(s)
└── <TICKER>_outlier_treated.csv   # same, after outlier treatment (adds Outlier_Treated)
charts/
├── <TICKER>_price_ema.png         # Close + EMA(s) with volume panel
└── <TICKER>_outlier_comparison.png# Close before vs after outlier treatment
```

Each CSV contains: `Datetime, Open, High, Low, Close, Volume, Adj Open, Adj High, Adj Low,
Adj Close, EMA_<n>…`. `Datetime` is timezone-aware (`America/New_York`) and covers regular
trading hours only.

## Notes

- **EMA** is the standard recursive exponential moving average
  (`Close.ewm(span=n, adjust=False)`), computed on the closing price.
- **"Live" data**: Yahoo Finance intraday bars are near-real-time (typically ~15 min
  delayed). For truly real-time data, point the same pipeline at a broker REST API.
- **Yahoo 5-minute limit**: Yahoo serves 5-minute data for only ~60 days, so `--days` is
  capped accordingly (`--max-intraday-days`).
- **Intraday x-axis**: charts use a gap-free bar axis with ticks at each trading-day
  boundary, so overnight/weekend gaps don't distort the price line.
- Outliers are treated on **intraday log-returns** with the **overnight gap excluded** (an
  overnight jump is a real gap, not a bad tick); EMAs are recomputed on the treated prices.
