# SPEC — Data Contracts Between Modules

This document defines the standard DataFrame/Series schemas passed between
`src/data`, `src/engine`, `src/backtest`, and `src/analysis`. Every module
must produce and consume these shapes exactly — no ad-hoc columns, no
silent reindexing. See [CLAUDE.md](CLAUDE.md) for the no-lookahead-bias rule
that governs how these objects may be transformed.

All time-indexed objects use a `pandas.DatetimeIndex`, UTC-aware, sorted
ascending, with no duplicate timestamps.

---

## 1. Price Data (`src/data` → `src/engine`)

### 1.1 Raw OHLCV (long format) — output of the fetch layer, cached as Parquet

Stored under `data/cache/<ticker>.parquet`.

| column      | dtype                  | notes                              |
|-------------|------------------------|-------------------------------------|
| `date`      | `datetime64[ns, UTC]`  | index                                |
| `open`      | `float64`               |                                      |
| `high`      | `float64`               |                                      |
| `low`       | `float64`               |                                      |
| `close`     | `float64`               |                                      |
| `adj_close` | `float64`               | dividend/split-adjusted             |
| `volume`    | `int64`                 |                                      |
| `ticker`    | `string`                | constant per file, kept for concat  |

- Index: `date` (`DatetimeIndex`), one row per trading day.
- No `NaN` in `adj_close` for any row — gaps must be dropped, not filled,
  by the fetch layer before caching.

### 1.2 Price Panel (wide format) — input to `src/engine`

Produced by `src/data` from one or more raw OHLCV frames.

- Index: `DatetimeIndex` (`date`), shared trading calendar across all
  tickers (inner join — no forward-filled prices across missing days).
- Columns: one per ticker, `dtype=float64`, values are `adj_close`.
- `df.columns.name == "ticker"`.

```
                 AAPL      MSFT
date
2024-01-02     185.64    370.87
2024-01-03     184.25    370.60
...
```

---

## 2. Spread / Signal Series (`src/engine` → `src/backtest`)

Output of cointegration + hedge ratio + z-score signal generation for a
single pair. Index matches the Price Panel index (or a suffix of it, once
warm-up windows are dropped).

| column         | dtype     | notes |
|----------------|-----------|-------|
| `spread`       | `float64` | `price_a - hedge_ratio * price_b` |
| `hedge_ratio`  | `float64` | rolling/estimated, **as of that date** |
| `zscore`       | `float64` | `(spread - rolling_mean) / rolling_std` |
| `signal`       | `int8`    | `{-1, 0, 1}`: short spread, flat, long spread |

**Lookahead-bias contract:** `hedge_ratio`, `zscore`, and `signal` at row
`t` must be computable using only data available up to and including `t`,
and `signal[t]` is the position to be *entered at t+1's open/close*
(never `t`'s own close). Any rolling statistic feeding `signal` must be
`.shift(1)`-ed relative to the spread it is evaluated against. `src/engine`
is responsible for this lag; `src/backtest` assumes it has already been
applied and trades `signal[t]` starting at bar `t+1`.

Metadata attached to the series/frame (as a `.attrs` dict, not columns):
`{"pair": ("AAPL", "MSFT"), "lookback": 60, "entry_z": 2.0, "exit_z": 0.5}`.

---

## 3. Trade Log (`src/backtest` → `src/analysis`)

One row per closed round-trip trade.

| column                | dtype                 | notes |
|-----------------------|------------------------|-------|
| `pair`                | `string`               | e.g. `"AAPL/MSFT"` |
| `entry_date`          | `datetime64[ns, UTC]`  | bar the position was opened |
| `exit_date`           | `datetime64[ns, UTC]`  | bar the position was closed |
| `side`                | `int8`                 | `1` = long spread, `-1` = short spread |
| `entry_price_a`       | `float64`              | |
| `entry_price_b`       | `float64`              | |
| `exit_price_a`        | `float64`              | |
| `exit_price_b`        | `float64`              | |
| `hedge_ratio`         | `float64`              | ratio used to size the trade |
| `qty_a`               | `float64`              | signed shares/units of leg A |
| `qty_b`               | `float64`              | signed shares/units of leg B |
| `transaction_costs`   | `float64`              | total costs, both legs, both fills |
| `gross_pnl`           | `float64`              | before costs |
| `net_pnl`             | `float64`              | after costs |
| `return_pct`          | `float64`              | net return on capital allocated to the trade |
| `holding_period_days` | `int64`                | |

The trade log must be accompanied by an **equity curve** DataFrame:

| column      | dtype                 | notes |
|-------------|------------------------|-------|
| `equity`    | `float64`              | mark-to-market portfolio value |
| `returns`   | `float64`              | period-over-period simple return |
| `drawdown`  | `float64`              | `equity / running_max(equity) - 1`, `<= 0` |

Index: `DatetimeIndex` matching the backtest's trading calendar
(every bar, not just trade dates).

---

## 4. Performance Metrics (`src/analysis` output)

A flat `dict[str, float]` (or single-row DataFrame for tear-sheet
rendering) with these required keys:

| key                    | type    | notes |
|------------------------|---------|-------|
| `total_return`         | `float` | |
| `annualized_return`    | `float` | |
| `annualized_volatility`| `float` | |
| `sharpe_ratio`         | `float` | annualized, using a stated risk-free rate |
| `sortino_ratio`        | `float` | |
| `max_drawdown`         | `float` | most negative value from the equity curve's `drawdown` column |
| `calmar_ratio`         | `float` | `annualized_return / abs(max_drawdown)` |
| `num_trades`           | `int`   | |
| `win_rate`             | `float` | fraction of trades with `net_pnl > 0` |
| `avg_trade_return`     | `float` | mean of `return_pct` across the trade log |
| `profit_factor`        | `float` | gross profit / gross loss |

Tear-sheet renderers in `src/analysis` consume the equity curve and trade
log directly; they must not recompute metrics using a different formula
than the ones registered here.

---

## Module Boundaries

- `src/data`: owns fetching (`yfinance`), Parquet caching under
  `data/cache/`, and producing Price Panels. Never computes signals.
- `src/engine`: owns cointegration testing, hedge ratio estimation, and
  z-score signal generation. Consumes Price Panels, produces Spread/Signal
  Series. Never simulates fills or PnL.
- `src/backtest`: owns portfolio simulation, order execution, and
  transaction cost modeling. Consumes Spread/Signal Series, produces Trade
  Logs and equity curves. Never computes summary statistics.
- `src/analysis`: owns performance metrics and tear-sheet generation.
  Consumes Trade Logs and equity curves only.
