# Assignment 6 — Multi-Indicator Algorithmic Trading Back-Test System

A Python algorithmic-trading **back-testing and performance-analysis system** built on **three
technical indicators — EMA crossover + MACD + RSI**. It downloads and cleans historical data,
generates Buy/Sell signals, simulates trades with **both P&L-based and indicator-based target
/ stop-loss exits**, and reports performance with a trade log, a summary report and charts.

## Strategy (long-only)

| Indicator | Role | Bullish condition |
|---|---|---|
| EMA crossover | trend | `EMA_fast > EMA_slow` |
| MACD | trend/momentum | `MACD > MACD_signal` |
| RSI (Wilder) | momentum filter | `rsi_buy_min ≤ RSI < rsi_overbought` |

- **Entry** (go long): all three bullish **simultaneously**.
- **Exit** — whichever triggers first (covers **both** required exit types):
  - **P&L stop loss**: `price ≤ entry × (1 − stop_pct)` (intrabar via Low)
  - **P&L target**: `price ≥ entry × (1 + target_pct)` (intrabar via High)
  - **Indicator stop**: `EMA_fast < EMA_slow` OR `MACD < MACD_signal`
  - **Indicator target**: `RSI ≥ rsi_overbought` (overbought → take profit)

**No look-ahead bias**: a signal on bar *t* is executed at the **open of bar t+1**; stop/target
are evaluated intrabar (High/Low); indicator exits act on the close.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Default: AAPL, 10 years, daily
python backtest_strategy.py

# Other examples
python backtest_strategy.py --ticker MSFT --years 5
python backtest_strategy.py --target-pct 0.15 --stop-pct 0.07
python backtest_strategy.py --ticker NVDA --rsi-overbought 75
```

Useful options: `--ticker`, `--years`, `--interval`, `--ema-fast`, `--ema-slow`,
`--rsi-period`, `--rsi-buy-min`, `--rsi-overbought`, `--target-pct`, `--stop-pct`,
`--initial-capital`, `--position-fraction`, `--cost-bps`. Run
`python backtest_strategy.py --help` for all options.

## Deliverables (produced by the script)

```
output/
├── <TICKER>_trades.csv          # Sample trade log — one row per round-trip trade
├── <TICKER>_summary_report.md   # Summary Report — performance analysis
└── <TICKER>_equity.csv          # per-bar equity curve
charts/
├── <TICKER>_strategy.png        # price + EMAs + Buy/Sell markers, MACD and RSI panels
└── <TICKER>_equity.png          # equity curve vs buy & hold, and drawdown
```

### Trade log columns
`trade_id, entry_date, exit_date, bars_held, entry_price, exit_price, shares, exit_reason,
pnl, return_pct, entry_rsi, exit_rsi, entry_macd_hist, exit_macd_hist,
ema_fast_gt_slow_at_exit, equity_after`

### Performance metrics (summary report)
Total return, CAGR, Sharpe (annualised), max drawdown, final equity, buy & hold benchmark,
number of trades, win rate, profit factor, average/avg win/avg loss P&L, expectancy, average
bars held, best/worst trade, and a breakdown of exit reasons.

## Data handling

- Prices are **split/dividend-adjusted** (`auto_adjust=True`), so the back-test is free of
  split artefacts.
- Cleaning removes duplicate dates, **fills missing values** (time interpolation +
  forward/backward fill) and drops invalid rows (non-positive prices, `High < Low`).

## Notes

- Costs: a per-side `--cost-bps` (default 1 bp) models commission + slippage on every fill.
- Position sizing: each entry invests `--position-fraction` of current equity (whole shares).
- Long-only, single instrument by design; parameters are fully configurable for experiments.
- Educational back-test — **not investment advice**. Past performance does not guarantee
  future results.
