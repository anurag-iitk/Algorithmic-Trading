"""Download, clean and store 10 years of EOD price-volume data for the top 50 S&P 500 stocks.

Assignment (Module 305): fetch and store historical price-volume data of multiple stocks.

What this script does
---------------------
1.  Downloads the last N years (default 10) of daily End-Of-Day (EOD) OHLCV data for the
    top 50 constituents of the S&P 500 index from Yahoo Finance (via the ``yfinance`` API).
2.  Cleans and processes each stock's data:
      * removes duplicate rows (same trading date),
      * handles missing values (time interpolation + forward/backward fill),
      * removes invalid rows (non-positive prices, High < Low),
      * derives fully adjusted OHLC prices from the adjustment factor (Adj Close / Close).
3.  Stores each cleaned stock in its own CSV file under ``data/cleaned/``.
4.  Treats outliers (robust MAD winsorisation of daily log-returns) and stores a second copy
    of every stock under ``data/outlier_treated/``.
5.  Writes a ``data/summary.csv`` audit report describing what happened to every ticker.

Usage
-----
    python fetch_stock_data.py                       # full run: top 50 stocks, 10 years
    python fetch_stock_data.py --top 5 --years 2     # quick subset for testing
    python fetch_stock_data.py --tickers AAPL,MSFT   # explicit tickers

Only the Python standard library plus ``yfinance``, ``pandas`` and ``numpy`` are required
(see requirements.txt).
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


# Top 50 S&P 500 constituents by index weight / market capitalisation.
# The index is float-cap weighted, so this list is the practical meaning of "top 50 stocks".
# Weights drift over time, so the list is configurable via --tickers / --tickers-file.
# (Compiled from public S&P 500 weightings; GOOGL/GOOG are the two Alphabet share classes.)
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

logger = logging.getLogger("fetch_stock_data")


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def download_history(ticker: str, start: str, end: str,
                     retries: int = 3, pause: float = 1.0) -> pd.DataFrame | None:
    """Download raw daily OHLCV data for a single ticker with simple retry/backoff.

    ``auto_adjust=False`` is used so that both the raw ``Close`` and the ``Adj Close``
    (adjusted for splits and dividends) columns are returned.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=False, actions=False, progress=False, threads=False,
            )
            if df is not None and not df.empty:
                return df
            logger.warning("%s: no data returned (attempt %d/%d)", ticker, attempt, retries)
        except Exception as exc:  # network / API errors -> retry
            logger.warning("%s: download error (attempt %d/%d): %s", ticker, attempt, retries, exc)
        time.sleep(pause * attempt)
    return None


def normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output to single-level OHLCV columns with a DatetimeIndex."""
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
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Clean a raw OHLCV frame and derive adjusted OHLC prices.

    Handles duplicates, missing values, invalid rows and computes adjusted prices.
    Returns the cleaned frame and a stats dictionary for the audit report.
    """
    stats: dict[str, object] = {"rows_raw": len(df)}
    df = df.sort_index()

    # --- duplicate trading dates ---
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
    df.index.name = "Date"
    stats["rows_clean"] = len(df)
    if len(df):
        stats["start_date"] = df.index.min().date().isoformat()
        stats["end_date"] = df.index.max().date().isoformat()
    return df, stats


# --------------------------------------------------------------------------------------
# Outlier treatment
# --------------------------------------------------------------------------------------
def treat_outliers(df: pd.DataFrame, threshold: float = 3.5) -> tuple[pd.DataFrame, int]:
    """Winsorise extreme daily log-returns using a robust MAD-based modified z-score.

    Outliers are detected on the daily log-returns of ``Adj Close`` (a near-stationary
    series, unlike price levels which legitimately trend over 10 years). Flagged returns
    are capped at the threshold and the price path is rebuilt from the capped returns; all
    price columns in the affected rows are scaled consistently so raw/adjusted stay aligned.
    An ``Outlier_Treated`` boolean column flags the rows that were modified.
    """
    treated = df.copy()
    log_ret = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    valid = log_ret.dropna()

    if len(valid) < 3:
        treated["Outlier_Treated"] = False
        return treated, 0

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0 or np.isnan(mad):
        treated["Outlier_Treated"] = False
        return treated, 0

    modified_z = 0.6745 * (log_ret - median) / mad
    outlier_mask = (modified_z.abs() > threshold).fillna(False)
    n_outliers = int(outlier_mask.sum())

    # Winsorise the returns and rebuild the adjusted-close path from them.
    cap = threshold * mad / 0.6745
    capped_ret = log_ret.clip(lower=median - cap, upper=median + cap)
    factors = np.exp(capped_ret.fillna(0.0))
    factors.iloc[0] = 1.0
    new_adj_close = df["Adj Close"].iloc[0] * factors.cumprod()

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
    df.to_csv(path, date_format="%Y-%m-%d", index=True)


def process_ticker(ticker: str, start: str, end: str, cleaned_dir: Path, treated_dir: Path,
                   threshold: float, retries: int, pause: float) -> dict:
    """Full pipeline for one ticker: download -> clean -> save -> treat outliers -> save."""
    result: dict[str, object] = {
        "ticker": ticker, "rows_raw": 0, "rows_clean": 0, "duplicates_removed": 0,
        "missing_filled": 0, "invalid_removed": 0, "outliers_treated": 0,
        "start_date": "", "end_date": "", "status": "OK",
    }

    raw = download_history(ticker, start, end, retries=retries, pause=pause)
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
    """Determine the ticker universe from CLI options, defaulting to the top 50."""
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        text = Path(args.tickers_file).read_text()
        return [line.strip().upper() for line in text.splitlines() if line.strip()]
    return SP500_TOP_50[: args.top]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, clean and store EOD price-volume data for the top S&P 500 stocks.",
    )
    parser.add_argument("--years", type=int, default=10, help="Years of history (default: 10).")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top S&P 500 stocks to use (default: 50).")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated tickers overriding the default top-50 list.")
    parser.add_argument("--tickers-file", type=str, default="",
                        help="Path to a file with one ticker per line.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory for output CSV files (default: data).")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Modified z-score threshold for outlier treatment (default: 3.5).")
    parser.add_argument("--retries", type=int, default=3, help="Download retries per ticker.")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Base seconds to pause between/after requests (politeness).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    start_str = start.strftime("%Y-%m-%d")
    end_str = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    tickers = resolve_tickers(args)
    output_dir = Path(args.output_dir)
    cleaned_dir = output_dir / "cleaned"
    treated_dir = output_dir / "outlier_treated"

    logger.info("Downloading %d ticker(s), %d years of EOD data (%s to %s).",
                len(tickers), args.years, start_str, end.strftime("%Y-%m-%d"))

    summaries: list[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%2d/%2d] %s ...", i, len(tickers), ticker)
        summary = process_ticker(
            ticker, start_str, end_str, cleaned_dir, treated_dir,
            threshold=args.threshold, retries=args.retries, pause=args.pause,
        )
        summaries.append(summary)
        if summary["status"] == "OK":
            logger.info(
                "        rows=%s  dupes=%s  filled=%s  invalid=%s  outliers=%s  (%s -> %s)",
                summary["rows_clean"], summary["duplicates_removed"], summary["missing_filled"],
                summary["invalid_removed"], summary["outliers_treated"],
                summary["start_date"], summary["end_date"],
            )
        else:
            logger.warning("        %s -> %s", ticker, summary["status"])
        time.sleep(args.pause)  # be polite to the API between tickers

    summary_df = pd.DataFrame(summaries, columns=[
        "ticker", "status", "rows_raw", "rows_clean", "duplicates_removed",
        "missing_filled", "invalid_removed", "outliers_treated", "start_date", "end_date",
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
