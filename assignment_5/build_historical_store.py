"""Build and incrementally update a historical intraday price-volume store for the top 50 S&P 500 stocks.

Assignment (Module 305): use a REST API to fetch and store intra-day price-volume data of
multiple stocks and build a *historical data store*.

What this script does
---------------------
1.  Fetches the current top-50 S&P 500 constituents **live** (ranked by index weight from
    slickcharts.com, with Wikipedia and a built-in list as fallbacks).
2.  For each stock, downloads intraday (default 5-minute) OHLCV bars from Yahoo Finance via
    the ``yfinance`` REST API, in date chunks.
3.  **Builds a persistent historical store**: it loads each stock's existing CSV, *merges*
    the freshly fetched bars into it (de-duplicating timestamps), and writes it back. The
    first run backfills the maximum available history; later runs only fetch a small recent
    window and append the new bars - so running this periodically grows the store over time.
4.  Cleans the data: removes duplicate timestamps, handles missing values (time interpolation
    + forward/backward fill), drops invalid rows, and derives adjusted OHLC prices.
5.  Stores each stock in its own CSV under ``data/cleaned/`` and a separate outlier-treated
    copy under ``data/outlier_treated/``.
6.  Writes a ``data/summary.csv`` audit report, including how many new bars each run added.

Important limitation (Yahoo Finance 5-minute data)
--------------------------------------------------
Yahoo Finance only serves intraday data at 5-minute resolution for roughly the **last 60
days**. The store therefore backfills at most ~60 days from Yahoo; run the script regularly
(e.g. via cron) to accumulate a longer history than Yahoo alone exposes. To backfill a
genuine multi-year intraday store in one shot, point the same pipeline at a broker REST API
(e.g. Alpaca / Polygon) and raise ``--max-intraday-days``.

Usage
-----
    python build_historical_store.py                     # update store for top 50 stocks
    python build_historical_store.py --top 3             # quick subset for testing
    python build_historical_store.py --full-refresh      # ignore existing store, re-backfill
    python build_historical_store.py --tickers AAPL,MSFT # explicit tickers

Only ``yfinance``, ``pandas``, ``numpy`` plus ``requests`` and ``lxml`` (for the live S&P 500
list) are required (see requirements.txt).
"""

from __future__ import annotations

import argparse
import io
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

# Exchange timezone for US equities; yfinance returns intraday bars in this timezone.
EXCHANGE_TZ = "America/New_York"

# Yahoo Finance only serves 5-minute (and finer) data for ~60 days. Stay just inside it.
YAHOO_INTRADAY_MAX_DAYS = 59

# Fallback list of the top 50 S&P 500 constituents by index weight, used only when the live
# list cannot be fetched (see fetch_sp500_top). Weights drift over time, hence the live fetch.
SP500_TOP_50: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "GOOG", "BRK-B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ",
    "WMT", "NFLX", "ABBV", "BAC", "CRM", "ORCL", "CVX", "KO", "MRK", "AMD",
    "PEP", "TMO", "LIN", "ADBE", "CSCO", "ACN", "MCD", "WFC", "ABT", "GE",
    "IBM", "DIS", "CAT", "PM", "QCOM", "GS", "TXN", "VZ", "INTU", "AXP",
]

# Canonical column order for the stored CSV files.
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Open", "Adj High", "Adj Low", "Adj Close"]
OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume",
                  "Adj Open", "Adj High", "Adj Low", "Adj Close"]

logger = logging.getLogger("build_historical_store")


# --------------------------------------------------------------------------------------
# S&P 500 constituent list (fetched live, ranked by index weight)
# --------------------------------------------------------------------------------------
SP500_SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; sp500-history-store/1.0)"}


def _normalize_symbol(symbol: str) -> str:
    """Map a display ticker to yfinance form (e.g. BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def _read_first_html_table(url: str) -> pd.DataFrame:
    """Fetch a URL with a browser-like User-Agent and parse its first HTML table."""
    import requests  # transitive dependency of yfinance; imported lazily
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))[0]


def fetch_sp500_from_slickcharts() -> list[str]:
    """Return all S&P 500 tickers ranked by index weight (descending), from slickcharts.com."""
    df = _read_first_html_table(SP500_SLICKCHARTS_URL)
    return [_normalize_symbol(s) for s in df["Symbol"].tolist()]


def fetch_sp500_from_wikipedia() -> list[str]:
    """Return S&P 500 constituent tickers (membership only, not weight-ranked), from Wikipedia."""
    df = _read_first_html_table(SP500_WIKIPEDIA_URL)
    return [_normalize_symbol(s) for s in df["Symbol"].tolist()]


def fetch_sp500_top(n: int, source: str = "auto") -> list[str]:
    """Return the top-``n`` S&P 500 tickers, fetching the live list where possible.

    ``source`` selects the provider: ``slickcharts`` (live, weight-ranked), ``wikipedia``
    (live membership, unranked), ``static`` (the built-in fallback list) or ``auto`` (try
    slickcharts, then wikipedia, then the static list). Any network/parsing failure falls
    through to the next option so the pipeline still runs offline.
    """
    if source == "static":
        return SP500_TOP_50[:n]

    providers = {
        "slickcharts": ("slickcharts.com (weight-ranked)", fetch_sp500_from_slickcharts),
        "wikipedia": ("Wikipedia (membership, unranked)", fetch_sp500_from_wikipedia),
    }
    order = ["slickcharts", "wikipedia"] if source == "auto" else [source]
    for key in order:
        label, fn = providers[key]
        try:
            tickers = fn()
        except Exception as exc:  # network / parsing failure -> try the next source
            logger.warning("Could not fetch S&P 500 list from %s: %s", key, exc)
            continue
        if tickers:
            if key == "wikipedia":
                logger.warning("Wikipedia's list is alphabetical, not weight-ranked; the first "
                               "%d names may not be the true top %d by index weight.", n, n)
            logger.info("Fetched %d S&P 500 constituents from %s.", len(tickers), label)
            return tickers[:n]
    logger.warning("Falling back to the built-in static top-50 list.")
    return SP500_TOP_50[:n]


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def _download_chunk(ticker: str, start: str, end: str, interval: str,
                    retries: int, pause: float) -> pd.DataFrame | None:
    """Download one date-range chunk of intraday OHLCV data with simple retry/backoff."""
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
# Historical store (load / merge / persist)
# --------------------------------------------------------------------------------------
def load_store(path: Path) -> pd.DataFrame | None:
    """Load an existing per-stock store CSV into a tz-aware frame, or None if absent/empty."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "Datetime" not in df.columns:
            return None
        # Parse via UTC to handle mixed DST offsets, then present in the exchange timezone.
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(EXCHANGE_TZ)
        df = df.set_index("Datetime").sort_index()
        return df
    except Exception as exc:  # corrupt/unreadable store -> treat as empty, will be rebuilt
        logger.warning("Could not read existing store %s: %s", path, exc)
        return None


def merge_store(existing: pd.DataFrame | None, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge freshly fetched rows into the existing store, de-duplicating by timestamp.

    Newer rows win on overlap (``keep='last'``) so revised bars replace older copies.
    Returns the merged frame and the count of genuinely new timestamps added.
    """
    if existing is None or existing.empty:
        return new, len(new)
    before = len(existing)
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    added = len(combined) - before
    return combined, max(added, 0)


def determine_fetch_start(existing: pd.DataFrame | None, today: pd.Timestamp,
                          backfill_days: int, overlap_days: int,
                          max_intraday_days: int, full_refresh: bool) -> pd.Timestamp:
    """Decide the download start date: backfill on first run, else a small recent overlap."""
    if existing is None or existing.empty or full_refresh:
        start = today - pd.Timedelta(days=backfill_days)
    else:
        last_ts = existing.index.max().tz_convert(EXCHANGE_TZ).normalize()
        start = last_ts - pd.Timedelta(days=overlap_days)
    # Never ask Yahoo for 5m data older than its ~60-day limit.
    return max(start, today - pd.Timedelta(days=max_intraday_days))


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean an intraday OHLCV frame and derive adjusted OHLC prices.

    Handles duplicates, missing values and invalid rows. Returns the cleaned frame and a
    stats dictionary for the audit report.
    """
    stats: dict[str, object] = {}
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
    df = df[OUTPUT_COLUMNS]
    df.index.name = "Datetime"
    return df, stats


# --------------------------------------------------------------------------------------
# Outlier treatment
# --------------------------------------------------------------------------------------
def treat_outliers(df: pd.DataFrame, threshold: float = 3.5) -> tuple[pd.DataFrame, int]:
    """Winsorise extreme intraday log-returns using a robust MAD-based modified z-score.

    Returns across the overnight gap (first bar of each session) are excluded - an overnight
    jump is a real gap, not a bad tick. Flagged returns are capped and each session's price
    path is rebuilt from its own first bar, so overnight gaps are preserved. An
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
        return treated, 0

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0 or np.isnan(mad):
        treated["Outlier_Treated"] = False
        return treated, 0

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
    return treated, n_outliers


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def update_ticker(ticker: str, today: pd.Timestamp, args: argparse.Namespace,
                  cleaned_dir: Path, treated_dir: Path) -> dict:
    """Incrementally update one stock's historical store, then its outlier-treated copy."""
    result: dict[str, object] = {
        "ticker": ticker, "status": "OK", "rows_before": 0, "rows_added": 0, "rows_total": 0,
        "duplicates_removed": 0, "missing_filled": 0, "invalid_removed": 0,
        "outliers_treated": 0, "trading_days": 0, "start": "", "end": "",
    }

    store_path = cleaned_dir / f"{ticker}.csv"
    existing = load_store(store_path)
    result["rows_before"] = 0 if existing is None else len(existing)

    start = determine_fetch_start(existing, today, args.backfill_days, args.overlap_days,
                                  args.max_intraday_days, args.full_refresh)
    end_exclusive = today + pd.Timedelta(days=1)

    raw = download_intraday(ticker, start, end_exclusive, args.interval,
                            args.chunk_days, args.retries, args.pause)
    if raw is None or raw.empty:
        # No new data (e.g. market closed). Keep the existing store as-is.
        if existing is not None and not existing.empty:
            result["status"] = "NO_NEW_DATA"
            result["rows_total"] = len(existing)
            return result
        result["status"] = "FAILED_DOWNLOAD"
        return result

    try:
        new_clean, stats = clean_data(normalize_columns(raw, ticker))
    except Exception as exc:
        logger.error("%s: cleaning failed: %s", ticker, exc)
        result["status"] = "FAILED_CLEAN"
        return result

    result.update(stats)
    merged, added = merge_store(existing, new_clean)
    if merged.empty:
        result["status"] = "EMPTY_AFTER_CLEAN"
        return result

    save_csv(merged, store_path)
    treated, n_outliers = treat_outliers(merged, threshold=args.threshold)
    save_csv(treated, treated_dir / f"{ticker}.csv")

    result["rows_added"] = added
    result["rows_total"] = len(merged)
    result["outliers_treated"] = n_outliers
    result["trading_days"] = int(pd.Series(merged.index.date).nunique())
    result["start"] = merged.index.min().isoformat()
    result["end"] = merged.index.max().isoformat()
    return result


def resolve_tickers(args: argparse.Namespace) -> list[str]:
    """Determine the ticker universe from CLI options, defaulting to the live top 50."""
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        text = Path(args.tickers_file).read_text()
        return [line.strip().upper() for line in text.splitlines() if line.strip()]
    return fetch_sp500_top(args.top, args.source)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/update a historical intraday store for the top S&P 500 stocks.",
    )
    parser.add_argument("--interval", type=str, default="5m",
                        help="Intraday bar size, e.g. 5m, 15m, 30m, 1h (default: 5m).")
    parser.add_argument("--backfill-days", type=int, default=YAHOO_INTRADAY_MAX_DAYS,
                        help="History to backfill on the first run / full refresh (default: %(default)s).")
    parser.add_argument("--overlap-days", type=int, default=3,
                        help="Recent window re-fetched on incremental runs to catch revisions "
                             "(default: 3).")
    parser.add_argument("--max-intraday-days", type=int, default=YAHOO_INTRADAY_MAX_DAYS,
                        help="Cap on the intraday lookback; Yahoo serves ~60 days of 5m data.")
    parser.add_argument("--chunk-days", type=int, default=30, help="Per-request chunk size (days).")
    parser.add_argument("--full-refresh", action="store_true",
                        help="Ignore the existing store and re-backfill from scratch.")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top S&P 500 stocks to use (default: 50).")
    parser.add_argument("--source", choices=["auto", "slickcharts", "wikipedia", "static"],
                        default="auto",
                        help="Where to get the S&P 500 list: auto (default; live weight-ranked "
                             "with fallbacks), slickcharts, wikipedia, or static.")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated tickers overriding the default top-50 list.")
    parser.add_argument("--tickers-file", type=str, default="",
                        help="Path to a file with one ticker per line.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory for the historical store (default: data).")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Modified z-score threshold for outlier treatment (default: 3.5).")
    parser.add_argument("--retries", type=int, default=3, help="Download retries per chunk.")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Base seconds to pause between requests (politeness).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    today = pd.Timestamp.today(tz=EXCHANGE_TZ).normalize()
    tickers = resolve_tickers(args)
    output_dir = Path(args.output_dir)
    cleaned_dir = output_dir / "cleaned"
    treated_dir = output_dir / "outlier_treated"

    mode = "full refresh" if args.full_refresh else "incremental update"
    logger.info("Historical store %s: %d ticker(s), %s bars, store dir '%s'.",
                mode, len(tickers), args.interval, output_dir)

    summaries: list[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%2d/%2d] %s ...", i, len(tickers), ticker)
        summary = update_ticker(ticker, today, args, cleaned_dir, treated_dir)
        summaries.append(summary)
        if summary["status"] == "OK":
            logger.info("        +%s new bars -> %s total  (days=%s, outliers=%s, %s -> %s)",
                        summary["rows_added"], summary["rows_total"], summary["trading_days"],
                        summary["outliers_treated"], summary["start"], summary["end"])
        elif summary["status"] == "NO_NEW_DATA":
            logger.info("        no new bars; store unchanged (%s rows)", summary["rows_total"])
        else:
            logger.warning("        %s -> %s", ticker, summary["status"])

    summary_df = pd.DataFrame(summaries, columns=[
        "ticker", "status", "rows_before", "rows_added", "rows_total", "duplicates_removed",
        "missing_filled", "invalid_removed", "outliers_treated", "trading_days", "start", "end",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    ok = summary_df[summary_df["status"].isin(["OK", "NO_NEW_DATA"])]
    failed = summary_df[~summary_df["status"].isin(["OK", "NO_NEW_DATA"])]
    total_added = int(summary_df["rows_added"].sum())
    logger.info("Done. %d/%d stores updated (+%d new bars total). Cleaned: %s  Treated: %s",
                len(ok), len(tickers), total_added, cleaned_dir, treated_dir)
    logger.info("Audit report written to %s", output_dir / "summary.csv")
    if not failed.empty:
        logger.warning("Failed tickers: %s", ", ".join(failed["ticker"].tolist()))
    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
