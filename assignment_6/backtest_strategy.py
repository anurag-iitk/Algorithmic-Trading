"""Multi-indicator algorithmic trading strategy back-tester (EMA + MACD + RSI).

Assignment (Module 305): develop a trading strategy from multiple technical indicators and a
back-testing / performance-analysis system in Python.

Strategy (long-only)
--------------------
Three complementary technical indicators drive the strategy:
  * EMA crossover (trend)        - fast EMA vs slow EMA
  * MACD (trend/momentum)        - MACD line vs its signal line
  * RSI (momentum, Wilder)       - avoids chasing overbought moves

Entry (go long) when ALL three agree bullish at the bar close:
  EMA_fast > EMA_slow  AND  MACD > MACD_signal  AND  rsi_buy_min <= RSI < rsi_overbought

Exit (close long) when ANY target/stop condition triggers - covering BOTH P&L terms and
indicator values, as required by the assignment:
  * P&L stop loss     : price <= entry * (1 - stop_pct)         (intrabar, via Low)
  * P&L target        : price >= entry * (1 + target_pct)       (intrabar, via High)
  * Indicator stop    : EMA_fast < EMA_slow  OR  MACD < MACD_signal   (trend/momentum lost)
  * Indicator target  : RSI >= rsi_overbought                   (overbought - take profit)

Execution avoids look-ahead bias: a signal on bar t is executed at the OPEN of bar t+1.
Stop/target are checked intrabar (High/Low); indicator exits act on the close.

Deliverables produced
---------------------
  * output/<TICKER>_trades.csv          - the sample trade log (one row per trade)
  * output/<TICKER>_summary_report.md   - the summary report (performance analysis)
  * output/<TICKER>_equity.csv          - the per-bar equity curve
  * charts/<TICKER>_strategy.png        - price + signals, MACD and RSI panels
  * charts/<TICKER>_equity.png          - equity curve vs buy & hold, and drawdown

Usage
-----
    python backtest_strategy.py                            # AAPL, 10y daily
    python backtest_strategy.py --ticker MSFT --years 5
    python backtest_strategy.py --target-pct 0.15 --stop-pct 0.07

Requires ``yfinance``, ``pandas``, ``numpy`` and ``matplotlib`` (see requirements.txt).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
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

# Approximate number of bars per year, for annualising returns / Sharpe by interval.
PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1h": 1638, "30m": 3276, "15m": 6552, "5m": 19656}

logger = logging.getLogger("backtest_strategy")


@dataclass
class StrategyParams:
    """All tunable strategy / back-test parameters."""
    ema_fast: int = 12
    ema_slow: int = 26
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_buy_min: float = 50.0
    rsi_overbought: float = 70.0
    target_pct: float = 0.10
    stop_pct: float = 0.05
    initial_capital: float = 100_000.0
    position_fraction: float = 0.95
    cost_bps: float = 1.0  # per-side cost (commission + slippage) in basis points


# --------------------------------------------------------------------------------------
# Data download & cleaning
# --------------------------------------------------------------------------------------
def download_prices(ticker: str, start: str, end: str, interval: str,
                    retries: int = 3, pause: float = 1.0) -> pd.DataFrame | None:
    """Download split/dividend-adjusted OHLCV data with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=True, actions=False, progress=False, threads=False,
            )
            if df is not None and not df.empty:
                return df
            logger.warning("%s: no data (attempt %d/%d)", ticker, attempt, retries)
        except Exception as exc:
            logger.warning("%s: download error (attempt %d/%d): %s", ticker, attempt, retries, exc)
        time.sleep(pause * attempt)
    return None


def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten columns, handle missing values and drop invalid rows (adjusted OHLCV)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        df.columns = df.columns.get_level_values(-1 if ticker in level0 else 0)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}")

    df = df[expected].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(how="all")

    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].interpolate(method="time", limit_direction="both").ffill().bfill()
    df["Volume"] = df["Volume"].ffill().fillna(0)

    df = df[(df[price_cols] > 0).all(axis=1) & (df["High"] >= df["Low"])]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's n-period RSI (SMA seed, then recursive smoothing)."""
    delta = close.diff().to_numpy(dtype="float64")
    n = len(delta)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        avg_gain[period] = gain[1:period + 1].mean()
        avg_loss[period] = loss[1:period + 1].mean()
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - (100.0 / (1.0 + rs))
    out[:period] = np.nan
    return pd.Series(out, index=close.index)


def add_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Attach EMA, MACD and RSI indicator columns to the price frame."""
    df = df.copy()
    df["EMA_fast"] = ema(df["Close"], p.ema_fast)
    df["EMA_slow"] = ema(df["Close"], p.ema_slow)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(
        df["Close"], p.macd_fast, p.macd_slow, p.macd_signal
    )
    df["RSI"] = rsi_wilder(df["Close"], p.rsi_period)
    return df


def generate_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Add boolean entry and indicator-exit columns from the three indicators."""
    df = df.copy()
    bull_ema = df["EMA_fast"] > df["EMA_slow"]
    bull_macd = df["MACD"] > df["MACD_signal"]
    bull_rsi = (df["RSI"] >= p.rsi_buy_min) & (df["RSI"] < p.rsi_overbought)
    df["entry_signal"] = bull_ema & bull_macd & bull_rsi

    # Indicator-based exit: trend/momentum lost, or overbought (take profit).
    df["indicator_exit"] = (
        (df["EMA_fast"] < df["EMA_slow"]) | (df["MACD"] < df["MACD_signal"])
        | (df["RSI"] >= p.rsi_overbought)
    )
    return df


# --------------------------------------------------------------------------------------
# Back-test engine
# --------------------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, p: StrategyParams) -> tuple[pd.DataFrame, pd.Series]:
    """Event-driven long-only back-test. Returns (trades, equity_curve).

    A signal on bar t is executed at the OPEN of bar t+1 (no look-ahead). Stop/target are
    evaluated intrabar via Low/High; indicator exits act on the close of the bar after entry.
    Any position still open at the end is closed at the final close.
    """
    indicator_cols = ["EMA_fast", "EMA_slow", "MACD", "MACD_signal", "RSI"]
    valid = df[indicator_cols].notna().all(axis=1) & df[["entry_signal", "indicator_exit"]].notna().all(axis=1)
    start = int(np.argmax(valid.to_numpy())) if valid.any() else len(df)

    dates = df.index
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    rsi = df["RSI"].to_numpy(float)
    macd_line = df["MACD"].to_numpy(float)
    macd_sig = df["MACD_signal"].to_numpy(float)
    ema_f = df["EMA_fast"].to_numpy(float)
    ema_s = df["EMA_slow"].to_numpy(float)
    entry_sig = df["entry_signal"].to_numpy(bool)
    indic_exit = df["indicator_exit"].to_numpy(bool)

    cost = p.cost_bps / 10_000.0
    n = len(df)
    cash = p.initial_capital
    position: dict | None = None
    pending_entry = False
    trades: list[dict] = []
    equity = np.full(n, p.initial_capital, dtype=float)

    for i in range(start, n):
        # 1) Execute a pending entry at this bar's open.
        if position is None and pending_entry:
            fill = o[i] * (1.0 + cost)
            shares = int((cash * p.position_fraction) // fill)
            if shares > 0:
                cash -= shares * fill
                position = {
                    "entry_index": i, "entry_date": dates[i], "entry_price": o[i],
                    "entry_fill": fill, "shares": shares,
                    "entry_rsi": rsi[i], "entry_macd": macd_line[i] - macd_sig[i],
                }
            pending_entry = False

        # 2) Manage an open position: stop/target intrabar, indicator exit on later bars.
        if position is not None:
            entry_price = position["entry_price"]
            stop_price = entry_price * (1.0 - p.stop_pct)
            target_price = entry_price * (1.0 + p.target_pct)
            exit_price = exit_reason = None

            if l[i] <= stop_price:
                exit_price, exit_reason = stop_price, "STOP_LOSS"
            elif h[i] >= target_price:
                exit_price, exit_reason = target_price, "TARGET"
            elif i > position["entry_index"] and indic_exit[i]:
                exit_price, exit_reason = c[i], "INDICATOR_EXIT"
            elif i == n - 1:
                exit_price, exit_reason = c[i], "END_OF_DATA"

            if exit_reason is not None:
                fill = exit_price * (1.0 - cost)
                shares = position["shares"]
                cash += shares * fill
                pnl = shares * (fill - position["entry_fill"])
                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_date": position["entry_date"], "exit_date": dates[i],
                    "bars_held": i - position["entry_index"],
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "shares": shares, "exit_reason": exit_reason,
                    "pnl": round(pnl, 2),
                    "return_pct": round((fill / position["entry_fill"] - 1.0) * 100.0, 4),
                    "entry_rsi": round(position["entry_rsi"], 2), "exit_rsi": round(rsi[i], 2),
                    "entry_macd_hist": round(position["entry_macd"], 4),
                    "exit_macd_hist": round(macd_line[i] - macd_sig[i], 4),
                    "ema_fast_gt_slow_at_exit": bool(ema_f[i] > ema_s[i]),
                    "equity_after": round(cash, 2),
                })
                position = None

        # 3) Generate a new entry signal (executed next bar) when flat.
        if position is None and not pending_entry and entry_sig[i]:
            pending_entry = True

        # 4) Mark-to-market equity at the close.
        equity[i] = cash + (position["shares"] * c[i] if position is not None else 0.0)

    equity_curve = pd.Series(equity[start:], index=dates[start:], name="equity")
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_curve


# --------------------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------------------
def compute_metrics(trades: pd.DataFrame, equity: pd.Series, prices: pd.DataFrame,
                    p: StrategyParams, interval: str) -> dict:
    """Compute strategy and benchmark performance metrics."""
    ppy = PERIODS_PER_YEAR.get(interval, 252)
    start_eq, end_eq = p.initial_capital, float(equity.iloc[-1]) if len(equity) else p.initial_capital
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9) if len(equity) > 1 else 1e-9

    rets = equity.pct_change().dropna()
    sharpe = float(np.sqrt(ppy) * rets.mean() / rets.std()) if rets.std() > 0 else float("nan")
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    # Benchmark: buy & hold over the back-tested window.
    bh_close = prices.loc[equity.index, "Close"] if len(equity) else prices["Close"]
    bh_return = float(bh_close.iloc[-1] / bh_close.iloc[0] - 1.0) if len(bh_close) > 1 else 0.0

    m: dict[str, object] = {
        "initial_capital": start_eq, "final_equity": end_eq,
        "total_return_pct": (end_eq / start_eq - 1.0) * 100.0,
        "cagr_pct": ((end_eq / start_eq) ** (1.0 / years) - 1.0) * 100.0,
        "buy_hold_return_pct": bh_return * 100.0,
        "sharpe": sharpe, "max_drawdown_pct": max_dd * 100.0,
        "years": years, "n_trades": int(len(trades)),
    }

    if len(trades):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        m.update({
            "win_rate_pct": len(wins) / len(trades) * 100.0,
            "avg_trade_pnl": float(trades["pnl"].mean()),
            "avg_win_pnl": float(wins["pnl"].mean()) if len(wins) else 0.0,
            "avg_loss_pnl": float(losses["pnl"].mean()) if len(losses) else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "expectancy": float(trades["pnl"].mean()),
            "avg_bars_held": float(trades["bars_held"].mean()),
            "best_trade_pnl": float(trades["pnl"].max()),
            "worst_trade_pnl": float(trades["pnl"].min()),
            "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
        })
    else:
        m.update({"win_rate_pct": 0.0, "avg_trade_pnl": 0.0, "avg_win_pnl": 0.0,
                  "avg_loss_pnl": 0.0, "profit_factor": float("nan"), "expectancy": 0.0,
                  "avg_bars_held": 0.0, "best_trade_pnl": 0.0, "worst_trade_pnl": 0.0,
                  "exit_reason_counts": {}})
    return m


# --------------------------------------------------------------------------------------
# Reporting & charts
# --------------------------------------------------------------------------------------
def write_summary_report(path: Path, ticker: str, interval: str, p: StrategyParams,
                         prices: pd.DataFrame, trades: pd.DataFrame, m: dict) -> None:
    """Write the Markdown summary report (performance analysis deliverable)."""
    span = f"{prices.index[0].date()} to {prices.index[-1].date()}"
    pf = m["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    exit_counts = ", ".join(f"{k}: {v}" for k, v in m["exit_reason_counts"].items()) or "n/a"
    verdict = "outperformed" if m["total_return_pct"] > m["buy_hold_return_pct"] else "underperformed"

    lines = [
        f"# Strategy Back-Test Summary Report - {ticker}",
        "",
        "## Strategy",
        "Long-only, three technical indicators: **EMA crossover + MACD + RSI**.",
        "",
        f"- **Entry**: EMA_fast > EMA_slow AND MACD > MACD_signal AND "
        f"{p.rsi_buy_min:.0f} <= RSI < {p.rsi_overbought:.0f}",
        f"- **Exit (P&L)**: stop loss -{p.stop_pct*100:.1f}% / target +{p.target_pct*100:.1f}%",
        f"- **Exit (indicators)**: EMA_fast < EMA_slow OR MACD < MACD_signal OR "
        f"RSI >= {p.rsi_overbought:.0f}",
        "",
        "## Data",
        f"- Ticker: **{ticker}**  |  Interval: **{interval}**  |  Period: **{span}** "
        f"({m['years']:.2f} years)",
        f"- Bars: {len(prices):,}  |  Prices: split/dividend-adjusted (auto_adjust)",
        f"- Costs: {p.cost_bps:.1f} bps per side  |  Position size: "
        f"{p.position_fraction*100:.0f}% of equity",
        "",
        "## Performance",
        "",
        "| Metric | Strategy | Buy & Hold |",
        "|---|--:|--:|",
        f"| Total return | {m['total_return_pct']:.2f}% | {m['buy_hold_return_pct']:.2f}% |",
        f"| CAGR | {m['cagr_pct']:.2f}% | - |",
        f"| Sharpe (ann.) | {m['sharpe']:.2f} | - |",
        f"| Max drawdown | {m['max_drawdown_pct']:.2f}% | - |",
        f"| Final equity | ${m['final_equity']:,.0f} | - |",
        "",
        "## Trade statistics",
        "",
        f"- Trades: **{m['n_trades']}**  |  Win rate: **{m['win_rate_pct']:.1f}%**  |  "
        f"Profit factor: **{pf_str}**",
        f"- Avg trade P&L: ${m['avg_trade_pnl']:,.2f}  |  Avg win: ${m['avg_win_pnl']:,.2f}  |  "
        f"Avg loss: ${m['avg_loss_pnl']:,.2f}",
        f"- Best trade: ${m['best_trade_pnl']:,.2f}  |  Worst trade: ${m['worst_trade_pnl']:,.2f}  |  "
        f"Avg bars held: {m['avg_bars_held']:.1f}",
        f"- Exit reasons: {exit_counts}",
        "",
        "## Verdict",
        f"The strategy **{verdict}** buy & hold on total return over this period "
        f"({m['total_return_pct']:.2f}% vs {m['buy_hold_return_pct']:.2f}%), with a Sharpe of "
        f"{m['sharpe']:.2f} and a max drawdown of {m['max_drawdown_pct']:.2f}%.",
        "",
        "_Generated by backtest_strategy.py. Educational back-test; not investment advice._",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _make_pyplot():
    try:
        import matplotlib
    except ImportError:  # pragma: no cover
        sys.exit("The 'matplotlib' package is required. Install with: pip install -r requirements.txt")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_strategy(df: pd.DataFrame, trades: pd.DataFrame, ticker: str, p: StrategyParams,
                  out_path: Path, plt) -> None:
    """Price with EMAs and buy/sell markers, plus MACD and RSI panels."""
    fig, (axp, axm, axr) = plt.subplots(
        3, 1, sharex=True, figsize=(14, 11), gridspec_kw={"height_ratios": [3, 1.3, 1.3]}
    )
    axp.plot(df.index, df["Close"], color="#1f1f1f", lw=1.0, label="Close (adj)")
    axp.plot(df.index, df["EMA_fast"], color="#4c78a8", lw=1.0, label=f"EMA-{p.ema_fast}")
    axp.plot(df.index, df["EMA_slow"], color="#f58518", lw=1.0, label=f"EMA-{p.ema_slow}")
    if len(trades):
        axp.scatter(trades["entry_date"], trades["entry_price"], marker="^",
                    color="#54a24b", s=70, zorder=5, label="Buy")
        axp.scatter(trades["exit_date"], trades["exit_price"], marker="v",
                    color="#e45756", s=70, zorder=5, label="Sell")
    axp.set_title(f"{ticker} - EMA+MACD+RSI strategy")
    axp.set_ylabel("Price (USD)")
    axp.legend(loc="best", ncol=2, fontsize=8)
    axp.grid(alpha=0.3)

    axm.plot(df.index, df["MACD"], color="#4c78a8", lw=1.0, label="MACD")
    axm.plot(df.index, df["MACD_signal"], color="#e45756", lw=1.0, label="Signal")
    axm.bar(df.index, df["MACD_hist"], color="#bbbbbb", width=1.0, label="Hist")
    axm.axhline(0, color="grey", lw=0.6)
    axm.set_ylabel("MACD")
    axm.legend(loc="best", ncol=3, fontsize=8)
    axm.grid(alpha=0.3)

    axr.plot(df.index, df["RSI"], color="#8434eb", lw=1.0)
    axr.axhline(p.rsi_overbought, color="#e45756", ls="--", lw=0.8)
    axr.axhline(p.rsi_buy_min, color="grey", ls=":", lw=0.6)
    axr.axhline(30, color="#54a24b", ls="--", lw=0.8)
    axr.set_ylim(0, 100)
    axr.set_yticks([0, 30, 50, 70, 100])
    axr.set_ylabel(f"RSI ({p.rsi_period})")
    axr.set_xlabel("Date")
    axr.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved strategy chart -> %s", out_path)


def plot_equity(equity: pd.Series, prices: pd.DataFrame, ticker: str, p: StrategyParams,
                out_path: Path, plt) -> None:
    """Equity curve vs buy & hold, with a drawdown panel."""
    bh_close = prices.loc[equity.index, "Close"]
    bh_equity = p.initial_capital * (bh_close / bh_close.iloc[0])
    drawdown = (equity / equity.cummax() - 1.0) * 100.0

    fig, (axe, axd) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    axe.plot(equity.index, equity, color="#4c78a8", lw=1.2, label="Strategy equity")
    axe.plot(bh_equity.index, bh_equity, color="#999999", lw=1.0, ls="--", label="Buy & hold")
    axe.set_title(f"{ticker} - equity curve vs buy & hold")
    axe.set_ylabel("Equity (USD)")
    axe.legend(loc="best")
    axe.grid(alpha=0.3)

    axd.fill_between(drawdown.index, drawdown, 0, color="#e45756", alpha=0.4)
    axd.set_ylabel("Drawdown (%)")
    axd.set_xlabel("Date")
    axd.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved equity chart -> %s", out_path)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back-test a 3-indicator (EMA+MACD+RSI) trading strategy.",
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock symbol (default: AAPL).")
    parser.add_argument("--years", type=float, default=10.0, help="Years of history (default: 10).")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Bar interval, e.g. 1d, 1wk, 1h (default: 1d).")
    parser.add_argument("--ema-fast", type=int, default=12)
    parser.add_argument("--ema-slow", type=int, default=26)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-buy-min", type=float, default=50.0)
    parser.add_argument("--rsi-overbought", type=float, default=70.0)
    parser.add_argument("--target-pct", type=float, default=0.10, help="Profit target (fraction).")
    parser.add_argument("--stop-pct", type=float, default=0.05, help="Stop loss (fraction).")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.95)
    parser.add_argument("--cost-bps", type=float, default=1.0, help="Per-side cost in bps.")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--charts-dir", type=str, default="charts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    ticker = args.ticker.strip().upper()
    p = StrategyParams(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow, macd_fast=args.ema_fast,
        macd_slow=args.ema_slow, rsi_period=args.rsi_period, rsi_buy_min=args.rsi_buy_min,
        rsi_overbought=args.rsi_overbought, target_pct=args.target_pct, stop_pct=args.stop_pct,
        initial_capital=args.initial_capital, position_fraction=args.position_fraction,
        cost_bps=args.cost_bps,
    )

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    logger.info("Downloading %s %s data (%s to %s) ...", ticker, args.interval,
                start.date(), end.date())
    raw = download_prices(ticker, start.strftime("%Y-%m-%d"),
                          (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), args.interval)
    if raw is None or raw.empty:
        logger.error("No data for %s.", ticker)
        return 1

    prices = clean_data(raw, ticker)
    df = generate_signals(add_indicators(prices, p), p)
    logger.info("Cleaned %d bars; running back-test ...", len(prices))

    trades, equity = run_backtest(df, p)
    if equity.empty:
        logger.error("Not enough data to back-test after indicator warm-up.")
        return 1
    m = compute_metrics(trades, equity, prices, p, args.interval)

    # --- deliverables ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / f"{ticker}_trades.csv"
    if len(trades):
        trades.to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["trade_id"]).to_csv(trades_path, index=False)
    equity.to_frame().to_csv(output_dir / f"{ticker}_equity.csv")
    write_summary_report(output_dir / f"{ticker}_summary_report.md", ticker, args.interval,
                         p, prices, trades, m)

    plt = _make_pyplot()
    charts_dir = Path(args.charts_dir)
    plot_strategy(df, trades, ticker, p, charts_dir / f"{ticker}_strategy.png", plt)
    plot_equity(equity, prices, ticker, p, charts_dir / f"{ticker}_equity.png", plt)

    # --- console summary ---
    pf = m["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    logger.info("=" * 68)
    logger.info("%s strategy: total return %.2f%% vs buy&hold %.2f%%  (CAGR %.2f%%)",
                ticker, m["total_return_pct"], m["buy_hold_return_pct"], m["cagr_pct"])
    logger.info("Trades: %d  win rate %.1f%%  profit factor %s  Sharpe %.2f  maxDD %.2f%%",
                m["n_trades"], m["win_rate_pct"], pf_str, m["sharpe"], m["max_drawdown_pct"])
    logger.info("Exit reasons: %s", m["exit_reason_counts"])
    logger.info("Deliverables -> %s , %s , %s",
                trades_path, output_dir / f"{ticker}_summary_report.md", charts_dir)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
