"""Download, clean and store intraday (5-minute) price-volume data for the top 50 S&P 500 stocks.

Assignment (Module 305): fetch and store *intra-day* historical price-volume data of multiple stocks.

What this script does
---------------------
1.  Fetches the current top-50 S&P 500 constituents **live** (ranked by index weight from
    slickcharts.com, with Wikipedia and a built-in list as fallbacks) and downloads intraday
    (default 5-minute) OHLCV bars for each from Yahoo Finance (via the ``yfinance`` API).
2.  Cleans and processes each stock's data:
      * removes duplicate rows (same timestamp),
      * handles missing values (time interpolation + forward/backward fill),
      * removes invalid rows (non-positive prices, High < Low),
      * derives fully adjusted OHLC prices from the adjustment factor (Adj Close / Close).
3.  Stores each cleaned stock in its own CSV file under ``data/cleaned/``.
4.  Treats outliers (robust MAD winsorisation of *intraday* log-returns, overnight gaps
    excluded) and stores a second copy of every stock under ``data/outlier_treated/``.
5.  Writes a ``data/summary.csv`` audit report describing what happened to every ticker.

Important limitation (Yahoo Finance 5-minute data)
--------------------------------------------------
The assignment asks for **1 year** of 5-minute data. Yahoo Finance only serves intraday
data at 5-minute resolution for roughly the **last 60 days** - a longer request is rejected
outright ("5m data not available ... must be within the last 60 days"). This script therefore
requests 1 year (``--days 365``) but automatically caps the intraday window to Yahoo's limit
(``--max-intraday-days``) and logs a clear warning. The download is done in date chunks
(``--chunk-days``); to obtain a genuine full year of intraday data, point the same pipeline at
a broker REST API (e.g. Alpaca / Polygon) that serves longer intraday history and raise
``--max-intraday-days`` - no other code changes are required.

Usage
-----
    python fetch_intraday_data.py                        # full run: top 50 stocks, 5-minute
    python fetch_intraday_data.py --top 3                # quick subset for testing
    python fetch_intraday_data.py --tickers AAPL,MSFT    # explicit tickers
    python fetch_intraday_data.py --interval 15m         # a different intraday resolution

Only the Python standard library plus ``yfinance``, ``pandas`` and ``numpy`` are required
(see requirements.txt).
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


# Fallback list of the top 50 S&P 500 constituents by index weight / market capitalisation,
# used only when the live list cannot be fetched (see fetch_sp500_top). The index is
# float-cap weighted, so this is the practical meaning of "top 50 stocks". Weights drift over
# time, hence the list is fetched live by default and is also overridable via --tickers.
# (GOOGL/GOOG are the two Alphabet share classes.)
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

# Yahoo Finance only serves 5-minute (and finer) data for ~60 days. Stay just inside it.
YAHOO_INTRADAY_MAX_DAYS = 59

logger = logging.getLogger("fetch_intraday_data")


# --------------------------------------------------------------------------------------
# S&P 500 constituent list (fetched live, ranked by index weight)
# --------------------------------------------------------------------------------------
SP500_SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; sp500-intraday-fetcher/1.0)"}


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
    """Download one date-range chunk of intraday OHLCV data with simple retry/backoff.

    ``auto_adjust=False`` is used so that both the raw ``Close`` and the ``Adj Close``
    (adjusted for splits and dividends) columns are returned. ``prepost=False`` keeps
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
            return None  # empty is a valid answer (e.g. window beyond Yahoo's cap)
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
    """Download intraday data over [start, end) in chunks and stitch the pieces together.

    Chunking keeps every request within the provider's per-request span limit and is the
    same technique required to page through long intraday history on a broker REST API.
    """
    frames: list[pd.DataFrame] = []
    for w_start, w_end in _chunk_ranges(start, end, chunk_days):
        chunk = _download_chunk(
            ticker, w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"),
            interval, retries, pause,
        )
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
        time.sleep(pause)  # be polite to the API between chunk requests

    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output to single-level OHLCV columns with a tz-aware DatetimeIndex."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) columns; keep the level holding the fields.
        level0 = set(df.columns.get_level_values(0))
        df.columns = df.columns.get_level_values(-1 if ticker in level0 else 0)

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}")

    df = df[expected]
    df.index = pd.to_datetime(df.index)  # keeps the exchange timezone for intraday bars
    df.index.name = "Datetime"
    return df


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Clean a raw intraday OHLCV frame and derive adjusted OHLC prices.

    Handles duplicates, missing values, invalid rows and computes adjusted prices.
    Returns the cleaned frame and a stats dictionary for the audit report.
    """
    stats: dict[str, object] = {"rows_raw": len(df)}
    df = df.sort_index()

    # --- duplicate timestamps ---
    dup_mask = df.index.duplicated(keep="first")
    stats["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    # --- drop fully empty rows, then handle remaining missing values ---
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

    # --- remove invalid rows (bad prints) ---
    invalid_mask = (df[price_cols] <= 0).any(axis=1) | (df["High"] < df["Low"])
    stats["invalid_removed"] = int(invalid_mask.sum())
    df = df[~invalid_mask]

    # --- adjusted prices: scale raw OHLC by the split/dividend adjustment factor ---
    factor = df["Adj Close"] / df["Close"]
    df["Adj Open"] = df["Open"] * factor
    df["Adj High"] = df["High"] * factor
    df["Adj Low"] = df["Low"] * factor

    df["Volume"] = df["Volume"].round().astype("int64")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].round(6)

    df = df[OUTPUT_COLUMNS]
    df.index.name = "Datetime"
    stats["rows_clean"] = len(df)
    if len(df):
        stats["trading_days"] = int(pd.Series(df.index.date).nunique())
        stats["start"] = df.index.min().isoformat()
        stats["end"] = df.index.max().isoformat()
    return df, stats


# --------------------------------------------------------------------------------------
# Outlier treatment
# --------------------------------------------------------------------------------------
def treat_outliers(df: pd.DataFrame, threshold: float = 3.5) -> tuple[pd.DataFrame, int]:
    """Winsorise extreme *intraday* log-returns using a robust MAD-based modified z-score.

    Outliers are detected on 5-minute log-returns of ``Adj Close``. Crucially, the return
    across the overnight gap (the first bar of each session) is excluded: an overnight jump
    is a real gap, not a bad tick, so it must not be flagged or altered. Flagged returns are
    capped at the threshold and each session's price path is rebuilt from its own first bar,
    so overnight gaps are preserved. All price columns in an affected row are scaled together
    so raw/adjusted stay aligned. An ``Outlier_Treated`` boolean column flags modified rows.
    """
    treated = df.copy()
    session = pd.Series(df.index.date, index=df.index)
    first_of_day = session.ne(session.shift(1))

    log_ret = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    intraday_ret = log_ret.where(~first_of_day)  # drop overnight returns
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

    # Winsorise intraday returns, then rebuild each session's path from its own first bar.
    cap = threshold * mad / 0.6745
    capped_ret = intraday_ret.clip(lower=median - cap, upper=median + cap)
    factors = np.exp(capped_ret)
    factors[first_of_day] = 1.0  # anchor each session at its actual first bar
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


def process_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str,
                   cleaned_dir: Path, treated_dir: Path, threshold: float,
                   chunk_days: int, retries: int, pause: float) -> dict:
    """Full pipeline for one ticker: download -> clean -> save -> treat outliers -> save."""
    result: dict[str, object] = {
        "ticker": ticker, "rows_raw": 0, "rows_clean": 0, "duplicates_removed": 0,
        "missing_filled": 0, "invalid_removed": 0, "outliers_treated": 0,
        "trading_days": 0, "start": "", "end": "", "status": "OK",
    }

    raw = download_intraday(ticker, start, end, interval, chunk_days, retries, pause)
    if raw is None or raw.empty:
        result["status"] = "FAILED_DOWNLOAD"
        return result

    try:
        cleaned, stats = clean_data(normalize_columns(raw, ticker), ticker)
    except Exception as exc:
        logger.error("%s: cleaning failed: %s", ticker, exc)
        result["status"] = "FAILED_CLEAN"
        return result

    result.update(stats)
    if cleaned.empty:
        result["status"] = "EMPTY_AFTER_CLEAN"
        return result

    save_csv(cleaned, cleaned_dir / f"{ticker}.csv")
    treated, n_outliers = treat_outliers(cleaned, threshold=threshold)
    save_csv(treated, treated_dir / f"{ticker}.csv")
    result["outliers_treated"] = n_outliers
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
        description="Download, clean and store intraday price-volume data for the top S&P 500 stocks.",
    )
    parser.add_argument("--days", type=int, default=365,
                        help="Requested days of history (default: 365 = 1 year).")
    parser.add_argument("--interval", type=str, default="5m",
                        help="Intraday bar size, e.g. 5m, 15m, 30m, 1h (default: 5m).")
    parser.add_argument("--max-intraday-days", type=int, default=YAHOO_INTRADAY_MAX_DAYS,
                        help="Cap on the intraday lookback window; Yahoo serves ~60 days of "
                             "5m data (default: %(default)s). Raise it for a broker API.")
    parser.add_argument("--chunk-days", type=int, default=30,
                        help="Per-request date-chunk size in days (default: 30).")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top S&P 500 stocks to use (default: 50).")
    parser.add_argument("--source", choices=["auto", "slickcharts", "wikipedia", "static"],
                        default="auto",
                        help="Where to get the S&P 500 list: auto (default; live weight-ranked "
                             "with fallbacks), slickcharts (live, weight-ranked), wikipedia "
                             "(live membership, unranked), or static (built-in list).")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated tickers overriding the default top-50 list.")
    parser.add_argument("--tickers-file", type=str, default="",
                        help="Path to a file with one ticker per line.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory for output CSV files (default: data).")
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

    end = pd.Timestamp.today().normalize()
    requested_start = end - pd.DateOffset(days=args.days)
    capped_start = end - pd.Timedelta(days=args.max_intraday_days)
    start = max(requested_start, capped_start)
    if start > requested_start:
        logger.warning(
            "Requested %d days of %s data, but Yahoo Finance only serves intraday data for "
            "~%d days. Capping the window to %s -> %s. For a genuine 1-year intraday history, "
            "use a broker REST API and raise --max-intraday-days.",
            args.days, args.interval, args.max_intraday_days,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        )

    end_exclusive = end + pd.Timedelta(days=1)  # include today's bars
    tickers = resolve_tickers(args)
    output_dir = Path(args.output_dir)
    cleaned_dir = output_dir / "cleaned"
    treated_dir = output_dir / "outlier_treated"

    logger.info("Downloading %d ticker(s), %s bars, %s to %s.",
                len(tickers), args.interval, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    summaries: list[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%2d/%2d] %s ...", i, len(tickers), ticker)
        summary = process_ticker(
            ticker, start, end_exclusive, args.interval, cleaned_dir, treated_dir,
            threshold=args.threshold, chunk_days=args.chunk_days,
            retries=args.retries, pause=args.pause,
        )
        summaries.append(summary)
        if summary["status"] == "OK":
            logger.info(
                "        rows=%s  days=%s  dupes=%s  filled=%s  invalid=%s  outliers=%s  (%s -> %s)",
                summary["rows_clean"], summary["trading_days"], summary["duplicates_removed"],
                summary["missing_filled"], summary["invalid_removed"], summary["outliers_treated"],
                summary["start"], summary["end"],
            )
        else:
            logger.warning("        %s -> %s", ticker, summary["status"])

    summary_df = pd.DataFrame(summaries, columns=[
        "ticker", "status", "rows_raw", "rows_clean", "duplicates_removed",
        "missing_filled", "invalid_removed", "outliers_treated", "trading_days", "start", "end",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    ok = summary_df[summary_df["status"] == "OK"]
    failed = summary_df[summary_df["status"] != "OK"]
    logger.info("Done. %d/%d succeeded. Cleaned files: %s  Treated files: %s",
                len(ok), len(tickers), cleaned_dir, treated_dir)
    logger.info("Audit report written to %s", output_dir / "summary.csv")
    if not failed.empty:
        logger.warning("Failed tickers: %s", ", ".join(failed["ticker"].tolist()))
    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
