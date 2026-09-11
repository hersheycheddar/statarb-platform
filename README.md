# Statistical Arbitrage Platform

A modular statistical arbitrage & pairs-trading backtesting engine: fetch
market data, screen for cointegrated pairs, backtest a mean-reversion
strategy, and report performance metrics.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
  data/loader.py       market data fetching + local Parquet caching
  engine/pairs.py       cointegration, hedge ratio, spread, z-score, half-life
  engine/screener.py    sector-wide cointegration screener
  backtest/engine.py    portfolio simulation / trade execution
  analysis/metrics.py   performance metrics (Sharpe, drawdown, etc.)
tests/                  pytest suite mirroring src/
run_cli.py              backtest a single pair end-to-end
run_screener.py         scan a sector (or ticker list) for tradable pairs
```

See [SPEC.md](SPEC.md) for the data contracts between modules and
[CLAUDE.md](CLAUDE.md) for the engineering rules (most importantly: no
lookahead bias).

## Usage

Backtest a single pair:

```bash
python run_cli.py --ticker-a KO --ticker-b PEP --lookback-years 3 --z-window 30
```

Screen a sector for cointegrated pairs:

```bash
python run_screener.py --sector energy --p-val 0.05 --max-half-life 45
```

Preset sectors: `energy`, `financials`, `etfs`, `tech_payments`, `dual_class`.
Or pass `--tickers KO,PEP,MNST` for a custom list.

## Testing

```bash
pytest
```
