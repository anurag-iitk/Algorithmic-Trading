"""Fetch intraday price-volume data for one stock, plot Close + EMA, and store CSVs.

Assignment (Module 305): use a REST API to fetch intra-day price-volume data of a stock,
plot a price chart and a technical indicator (EMA), handle missing data, and store the data
(plus an outlier-treated copy) as CSV.

What this script does
---------------------
1.  Fetches intraday (default 5-minute) OHLCV bars for a single stock (default AAPL) from
    Yahoo Finance via the ``yfinance`` REST API, downloaded in date chunks.
2.  Cleans the data: removes duplicate timestamps, handles missing values (time interpolation
    + forward/backward fill), drops invalid rows, and derives adjusted OHLC prices.
3.  Computes the running n-period EMA(s) of the closing price (default 12-period).
4.  Plots the closing prices and the EMA(s) on a price chart (with a volume panel) and saves
    it as a PNG; also saves a chart comparing prices before/after outlier treatment.
5.  Stores the cleaned data as CSV, and a separate outlier-treated copy as CSV.

Usage
-----
    python fetch_plot_indicators.py                          # AAPL, 5m, EMA-12, last 5 days
    python fetch_plot_indicators.py --ticker MSFT --ema 12,26
    python fetch_plot_indicators.py --interval 15m --days 30
    python fetch_plot_indicators.py --show                   # also display the charts

Note on "live" data: Yahoo Finance intraday bars are near-real-time (typically ~15 min
delayed) and cover regular trading hours. For truly real-time data use a broker REST API.

Requires ``yfinance``, ``pandas``, ``numpy`` and ``matplotlib`` (see requirements.txt).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled at runtime
    sys.exit(
        "The 'yfinance' package is required. Install dependencies with:\n"
        "    pip install -r requirements.txt"
    )

# Yahoo Finance only serves 5-minute (and finer) data for ~60 days. Stay just inside it.
YAHOO_INTRADAY_MAX_DAYS = 59

# Canonical price column order (adjusted prices are derived during cleaning).
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Open", "Adj High", "Adj Low", "Adj Close"]
BASE_OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume",
                       "Adj Open", "Adj High", "Adj Low", "Adj Close"]

logger = logging.getLogger("fetch_plot_indicators")


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def _download_chunk(ticker: str, start: str, end: str, interval: str,
                    retries: int, pause: float) -> pd.DataFrame | None:
    """Download one date-range chunk of intraday OHLCV data with simple retry/backoff.

    ``auto_adjust=False`` keeps both raw ``Close`` and ``Adj Close``; ``prepost=False`` keeps
    regular-trading-hours bars only.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=False, actions=False, prepost=False,
                progress=False, threads=False,
            )
            if df is not None and not df.empty:
                return df
            return None
        except Exception as exc:  # network / API errors -> retry
            logger.warning("%s: download error (attempt %d/%d): %s", ticker, attempt, retries, exc)
            time.sleep(pause * attempt)
    return None


def _chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int):
    """Yield [start, end) sub-ranges of at most ``chunk_days`` days spanning [start, end)."""
    cursor = start
    step = pd.Timedelta(days=chunk_days)
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def download_intraday(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str,
                      chunk_days: int, retries: int, pause: float) -> pd.DataFrame | None:
    """Download intraday data over [start, end) in chunks and stitch the pieces together."""
    frames: list[pd.DataFrame] = []
    for w_start, w_end in _chunk_ranges(start, end, chunk_days):
        chunk = _download_chunk(
            ticker, w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"),
            interval, retries, pause,
        )
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
        time.sleep(pause)

    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output to single-level OHLCV columns with a tz-aware DatetimeIndex."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        df.columns = df.columns.get_level_values(-1 if ticker in level0 else 0)

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}")

    df = df[expected]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Datetime"
    return df


# --------------------------------------------------------------------------------------
# Cleaning & indicators
# --------------------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a raw intraday OHLCV frame and derive adjusted OHLC prices.

    Handles duplicates, missing values and invalid rows. Returns the cleaned frame and a
    stats dict.
    """
    stats: dict[str, object] = {"rows_raw": len(df)}
    df = df.sort_index()

    dup_mask = df.index.duplicated(keep="first")
    stats["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    df = df.dropna(how="all")
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    stats["missing_filled"] = int(df[price_cols].isna().sum().sum())

    df[price_cols] = (
        df[price_cols]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    df["Volume"] = df["Volume"].ffill().fillna(0)
    df = df.dropna(subset=["Close", "Adj Close"])

    invalid_mask = (df[price_cols] <= 0).any(axis=1) | (df["High"] < df["Low"])
    stats["invalid_removed"] = int(invalid_mask.sum())
    df = df[~invalid_mask]

    factor = df["Adj Close"] / df["Close"]
    df["Adj Open"] = df["Open"] * factor
    df["Adj High"] = df["High"] * factor
    df["Adj Low"] = df["Low"] * factor

    df["Volume"] = df["Volume"].round().astype("int64")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].round(6)
    df = df[BASE_OUTPUT_COLUMNS]
    df.index.name = "Datetime"

    stats["rows_clean"] = len(df)
    if len(df):
        stats["trading_days"] = int(pd.Series(df.index.date).nunique())
    return df, stats


def add_ema(df: pd.DataFrame, periods: list[int], price_col: str = "Close") -> pd.DataFrame:
    """Add running n-period EMA column(s) of ``price_col`` (standard recursive EMA)."""
    df = df.copy()
    for n in periods:
        df[f"EMA_{n}"] = df[price_col].ewm(span=n, adjust=False).mean().round(6)
    return df


# --------------------------------------------------------------------------------------
# Outlier treatment
# --------------------------------------------------------------------------------------
def treat_outliers(df: pd.DataFrame, ema_periods: list[int],
                   threshold: float = 3.5) -> tuple[pd.DataFrame, int]:
    """Winsorise extreme intraday log-returns using a robust MAD-based modified z-score.

    Returns across the overnight gap (first bar of each session) are excluded - an overnight
    jump is a real gap, not a bad tick. Flagged returns are capped and each session's price
    path is rebuilt from its own first bar. EMAs are recomputed on the treated prices. An
    ``Outlier_Treated`` boolean column flags modified rows.
    """
    treated = df.copy()
    session = pd.Series(df.index.date, index=df.index)
    first_of_day = session.ne(session.shift(1))

    log_ret = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    intraday_ret = log_ret.where(~first_of_day)
    valid = intraday_ret.dropna()

    if len(valid) < 3:
        treated["Outlier_Treated"] = False
        return add_ema(treated, ema_periods), 0

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0 or np.isnan(mad):
        treated["Outlier_Treated"] = False
        return add_ema(treated, ema_periods), 0

    modified_z = 0.6745 * (intraday_ret - median) / mad
    outlier_mask = (modified_z.abs() > threshold).fillna(False)
    n_outliers = int(outlier_mask.sum())

    cap = threshold * mad / 0.6745
    capped_ret = intraday_ret.clip(lower=median - cap, upper=median + cap)
    factors = np.exp(capped_ret)
    factors[first_of_day] = 1.0
    day_first_adj = df["Adj Close"].groupby(session).transform("first")
    new_adj_close = day_first_adj * factors.groupby(session).cumprod()

    scale = new_adj_close / df["Adj Close"]
    for col in PRICE_COLUMNS:
        treated[col] = (df[col] * scale).round(6)

    treated["Outlier_Treated"] = outlier_mask.to_numpy()
    treated = add_ema(treated, ema_periods)  # recompute EMA on treated prices
    return treated, n_outliers


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def _make_pyplot(show: bool):
    """Import pyplot with a headless (Agg) backend unless interactive display is requested."""
    try:
        import matplotlib
    except ImportError:  # pragma: no cover
        sys.exit("The 'matplotlib' package is required. Install with: pip install -r requirements.txt")
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _session_ticks(index: pd.DatetimeIndex):
    """Return x positions and labels at each trading-day boundary (gap-free intraday axis)."""
    dates = pd.Series(index.date, index=index)
    boundaries = np.where(dates.ne(dates.shift(1)).to_numpy())[0]
    labels = [index[i].strftime("%Y-%m-%d") for i in boundaries]
    return boundaries, labels


def plot_price_ema(df: pd.DataFrame, ticker: str, ema_periods: list[int], interval: str,
                   out_path: Path, plt, show: bool = False) -> None:
    """Plot closing price + EMA(s) with a volume panel, using a gap-free intraday x-axis."""
    x = np.arange(len(df))
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(x, df["Close"], color="#1f1f1f", linewidth=1.0, label="Close")
    for n in ema_periods:
        ax_price.plot(x, df[f"EMA_{n}"], linewidth=1.4, label=f"EMA-{n}")
    ax_price.set_title(f"{ticker} - {interval} closing price & EMA")
    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(loc="best")
    ax_price.grid(alpha=0.3)

    ax_vol.bar(x, df["Volume"], color="#4c78a8", width=1.0)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.3)

    ticks, labels = _session_ticks(df.index)
    ax_vol.set_xticks(ticks)
    ax_vol.set_xticklabels(labels, rotation=45, ha="right")
    ax_vol.set_xlabel("Trading session (intraday bars, overnight gaps removed)")
    ax_vol.set_xlim(0, len(df) - 1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    logger.info("Saved price/EMA chart -> %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_outlier_comparison(cleaned: pd.DataFrame, treated: pd.DataFrame, ticker: str,
                            out_path: Path, plt, show: bool = False) -> None:
    """Plot cleaned vs outlier-treated closing prices, highlighting the treated bars."""
    x = np.arange(len(cleaned))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, cleaned["Close"], color="#4c78a8", linewidth=1.0, label="Close (cleaned)")
    ax.plot(x, treated["Close"], color="#f58518", linewidth=1.0, label="Close (outlier-treated)")

    flagged = treated["Outlier_Treated"].to_numpy()
    if flagged.any():
        ax.scatter(x[flagged], cleaned["Close"].to_numpy()[flagged],
                   color="#e45756", s=18, zorder=5, label="Detected outliers")

    ax.set_title(f"{ticker} - closing price before vs after outlier treatment")
    ax.set_ylabel("Price (USD)")
    ticks, labels = _session_ticks(cleaned.index)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Trading session (intraday bars, overnight gaps removed)")
    ax.set_xlim(0, len(cleaned) - 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    logger.info("Saved outlier-comparison chart -> %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def parse_ema_periods(text: str) -> list[int]:
    """Parse a comma-separated list of EMA periods, e.g. '12,26' -> [12, 26]."""
    periods = []
    for part in text.split(","):
        part = part.strip()
        if part:
            periods.append(int(part))
    if not periods:
        raise argparse.ArgumentTypeError("at least one EMA period is required")
    return periods


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch intraday data for one stock, plot Close + EMA, and store CSVs.",
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock symbol (default: AAPL).")
    parser.add_argument("--interval", type=str, default="5m",
                        help="Intraday bar size, e.g. 5m, 15m, 30m, 1h (default: 5m).")
    parser.add_argument("--days", type=int, default=5,
                        help="Intraday lookback window in days (default: 5).")
    parser.add_argument("--ema", type=parse_ema_periods, default="12",
                        help="Comma-separated EMA period(s) of the close (default: 12).")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Modified z-score threshold for outlier treatment (default: 3.5).")
    parser.add_argument("--max-intraday-days", type=int, default=YAHOO_INTRADAY_MAX_DAYS,
                        help="Cap on the intraday lookback; Yahoo serves ~60 days of 5m data.")
    parser.add_argument("--chunk-days", type=int, default=30, help="Per-request chunk size (days).")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory for CSV output.")
    parser.add_argument("--charts-dir", type=str, default="charts", help="Directory for PNG charts.")
    parser.add_argument("--retries", type=int, default=3, help="Download retries per chunk.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to pause between requests.")
    parser.add_argument("--show", action="store_true", help="Display the charts interactively.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    ticker = args.ticker.strip().upper()
    ema_periods = args.ema

    end = pd.Timestamp.today().normalize()
    requested_start = end - pd.Timedelta(days=args.days)
    capped_start = end - pd.Timedelta(days=args.max_intraday_days)
    start = max(requested_start, capped_start)
    if start > requested_start:
        logger.warning(
            "Requested %d days of %s data, but Yahoo serves intraday data for ~%d days; "
            "capping to %s.", args.days, args.interval, args.max_intraday_days,
            start.strftime("%Y-%m-%d"),
        )
    end_exclusive = end + pd.Timedelta(days=1)

    logger.info("Fetching %s %s bars from %s to %s ...",
                ticker, args.interval, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    raw = download_intraday(ticker, start, end_exclusive, args.interval,
                            args.chunk_days, args.retries, args.pause)
    if raw is None or raw.empty:
        logger.error("No data returned for %s. Check the symbol / interval / network.", ticker)
        return 1

    cleaned, stats = clean_data(normalize_columns(raw, ticker))
    if cleaned.empty:
        logger.error("%s: no rows left after cleaning.", ticker)
        return 1
    cleaned = add_ema(cleaned, ema_periods)
    logger.info("Cleaned: rows=%s  days=%s  dupes=%s  filled=%s  invalid=%s",
                stats["rows_clean"], stats.get("trading_days", 0), stats["duplicates_removed"],
                stats["missing_filled"], stats["invalid_removed"])

    treated, n_outliers = treat_outliers(cleaned, ema_periods, threshold=args.threshold)
    logger.info("Outliers treated: %d", n_outliers)

    # --- store CSVs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv = output_dir / f"{ticker}_cleaned.csv"
    treated_csv = output_dir / f"{ticker}_outlier_treated.csv"
    cleaned.to_csv(cleaned_csv, index=True)
    treated.to_csv(treated_csv, index=True)
    logger.info("Saved CSVs -> %s , %s", cleaned_csv, treated_csv)

    # --- plot charts ---
    plt = _make_pyplot(args.show)
    charts_dir = Path(args.charts_dir)
    plot_price_ema(cleaned, ticker, ema_periods, args.interval,
                   charts_dir / f"{ticker}_price_ema.png", plt, show=args.show)
    plot_outlier_comparison(cleaned, treated, ticker,
                            charts_dir / f"{ticker}_outlier_comparison.png", plt, show=args.show)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
