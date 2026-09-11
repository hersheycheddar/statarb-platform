#!/usr/bin/env python3
"""End-to-end pairs-trading run: fetch data, analyze, backtest, report.

Usage:
    python run_cli.py
    python run_cli.py --ticker-a COP --ticker-b SLB --lookback-years 3 --z-window 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from src.analysis.metrics import calculate_metrics, format_metrics_summary
from src.backtest.engine import PairsBacktester
from src.data.loader import MarketDataLoader
from src.engine.pairs import PairsAnalyzer


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch, analyze, and backtest a single cointegrated pair."
    )
    parser.add_argument("--ticker-a", type=str, default="KO", dest="ticker_a")
    parser.add_argument("--ticker-b", type=str, default="PEP", dest="ticker_b")
    parser.add_argument("--lookback-years", type=float, default=3.0, dest="lookback_years")
    parser.add_argument("--z-window", type=int, default=30, dest="z_window")
    return parser.parse_args(argv)


def _print_header(title: str, width: int = 60) -> None:
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def _print_trade_log_tail(trade_log: pd.DataFrame, n: int = 5) -> None:
    if trade_log.empty:
        print("No trades were executed.")
        return
    tail = trade_log.tail(n)
    header = (
        f"{'Entry':<12}{'Exit':<12}{'Type':<7}{'Entry A':>10}{'Entry B':>10}"
        f"{'Exit A':>10}{'Exit B':>10}{'Net P&L':>12}"
    )
    print(header)
    print("-" * len(header))
    for _, trade in tail.iterrows():
        print(
            f"{trade['entry_date'].strftime('%Y-%m-%d'):<12}"
            f"{trade['exit_date'].strftime('%Y-%m-%d'):<12}"
            f"{trade['type']:<7}"
            f"{trade['entry_price_a']:>10.2f}"
            f"{trade['entry_price_b']:>10.2f}"
            f"{trade['exit_price_a']:>10.2f}"
            f"{trade['exit_price_b']:>10.2f}"
            f"{trade['net_pnl']:>12.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    ticker_a = args.ticker_a.upper()
    ticker_b = args.ticker_b.upper()
    z_window = args.z_window

    end = date.today()
    start = end - timedelta(days=int(365 * args.lookback_years))

    _print_header("STATISTICAL ARBITRAGE PAIRS BACKTEST")
    print(f"Pair:     {ticker_a} / {ticker_b}")
    print(f"Period:   {start} to {end}")
    print(f"Z window: {z_window} days")
    print()

    loader = MarketDataLoader([ticker_a, ticker_b], start=start, end=end)
    prices = loader.fetch()
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]
    print(f"Loaded {len(prices)} aligned trading days.")
    print()

    analyzer = PairsAnalyzer()
    coint_result = analyzer.test_cointegration(price_a, price_b)
    hedge_ratio = analyzer.calculate_hedge_ratio(price_a, price_b, z_window)
    spread = analyzer.calculate_spread(price_a, price_b, hedge_ratio)
    zscore = analyzer.calculate_zscore(spread, z_window)
    half_life = analyzer.calculate_half_life(spread.dropna())

    print("COINTEGRATION")
    print(f"  Engle-Granger t-stat:   {coint_result.t_statistic:.4f}")
    print(f"  p-value:                {coint_result.p_value:.4f}")
    print(
        "  Critical values:        "
        + ", ".join(f"{k}: {v:.4f}" for k, v in coint_result.critical_values.items())
    )
    print(f"  Est. half-life (days):  {half_life:.1f}")
    print(f"  Latest hedge ratio:     {hedge_ratio.dropna().iloc[-1]:.4f}")
    print()

    backtester = PairsBacktester()
    portfolio, trade_log = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

    if portfolio["returns"].dropna().empty:
        print("Not enough data survived warm-up to run the backtest.")
        return 1

    metrics = calculate_metrics(portfolio, trade_log)

    print(format_metrics_summary(metrics))
    print()

    print("LAST 5 TRADES")
    _print_trade_log_tail(trade_log, 5)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
