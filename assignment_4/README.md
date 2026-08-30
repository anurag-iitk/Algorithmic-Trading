# Assignment 4 — Intraday Price Chart with RSI Technical Indicator

Python program that uses the **Yahoo Finance REST API** (via `yfinance`) to fetch **intraday
5-minute price-volume data** for **a single stock**, computes and **plots the running
n-period RSI**, handles missing data, and stores the data (plus an outlier-treated copy) as
CSV.

## What it does

1. **Fetch** — intraday (default 5-minute) OHLCV bars for one stock (default `AAPL`) from
   Yahoo Finance, downloaded in date chunks.
2. **Clean / handle missing data** — removes duplicate timestamps, fills missing values
   (time interpolation + forward/backward fill), drops invalid rows, and derives **adjusted**
   OHLC prices.
3. **Technical indicator** — computes the running **n-period RSI** of the closing price using
   **Wilder's smoothing** (default **14-period**).
4. **Plot** — a price chart with a **volume panel** and an **RSI panel** (with 30/70
   oversold/overbought bands), saved as a PNG, plus a chart comparing prices **before vs
   after outlier treatment**.
5. **Store** — the cleaned data as CSV, and a **separate outlier-treated copy** as CSV.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Default: AAPL, 5-minute bars, RSI-14, last 5 days
python fetch_plot_rsi.py

# Other examples
python fetch_plot_rsi.py --ticker MSFT --rsi 14
python fetch_plot_rsi.py --interval 15m --days 30
python fetch_plot_rsi.py --show          # also display the charts interactively
```

Useful options: `--ticker`, `--interval`, `--days`, `--rsi`, `--threshold` (outlier
sensitivity), `--output-dir`, `--charts-dir`, `--show`. Run
`python fetch_plot_rsi.py --help` for all options.

## Output layout

```
data/
├── <TICKER>_cleaned.csv           # cleaned OHLCV + adjusted prices + RSI column
└── <TICKER>_outlier_treated.csv   # same, after outlier treatment (adds Outlier_Treated)
charts/
├── <TICKER>_price_rsi.png         # price + volume + RSI panels
└── <TICKER>_outlier_comparison.png# Close before vs after outlier treatment
```

Each CSV contains: `Datetime, Open, High, Low, Close, Volume, Adj Open, Adj High, Adj Low,
Adj Close, RSI_<n>`. `Datetime` is timezone-aware (`America/New_York`) and covers regular
trading hours only.

## How RSI is computed (Wilder's method)

```
delta      = Close.diff()
gain       = max(delta, 0);  loss = max(-delta, 0)
avg_gain[n] = mean(gain[1..n]);  avg_loss[n] = mean(loss[1..n])          # SMA seed
avg_gain[i] = (avg_gain[i-1]*(n-1) + gain[i]) / n   (i > n)              # Wilder smoothing
RS  = avg_gain / avg_loss
RSI = 100 - 100 / (1 + RS)
```

The first `n` values are `NaN` (insufficient history). A window with no losses gives RSI 100;
RSI > 70 is conventionally "overbought" and RSI < 30 "oversold".

## Notes

- **"Live" data**: Yahoo Finance intraday bars are near-real-time (typically ~15 min
  delayed). For truly real-time data, point the same pipeline at a broker REST API.
- **Yahoo 5-minute limit**: Yahoo serves 5-minute data for only ~60 days, so `--days` is
  capped accordingly (`--max-intraday-days`).
- **Intraday x-axis**: charts use a gap-free bar axis with ticks at each trading-day
  boundary, so overnight/weekend gaps don't distort the price line.
- Outliers are treated on **intraday log-returns** with the **overnight gap excluded** (an
  overnight jump is a real gap, not a bad tick); RSI is recomputed on the treated prices.
