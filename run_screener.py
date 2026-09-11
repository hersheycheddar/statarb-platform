#!/usr/bin/env python3
"""Scan a sector (or custom ticker list) for statistically valid cointegrated pairs.

Usage:
    python run_screener.py --sector energy
    python run_screener.py --tickers KO,PEP,MNST,KDP --p-val 0.05 --max-half-life 60
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.engine.screener import SECTOR_UNIVERSES, PairScreener

SECTOR_ALIASES: dict[str, str] = {
    "energy": "ENERGY",
    "financials": "FINANCIALS",
    "etfs": "COMMODITY_ETFS",
    "commodities": "COMMODITY_ETFS",
    "commodity_etfs": "COMMODITY_ETFS",
    "tech_payments": "TECH_PAYMENTS",
    "payments": "TECH_PAYMENTS",
    "dual_class": "DUAL_CLASS",
}

REPORTS_DIR = Path("reports")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen a sector or custom ticker universe for cointegrated pairs."
    )
    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument(
        "--sector",
        type=str,
        help=f"Preset sector universe: {', '.join(sorted(SECTOR_ALIASES))}",
    )
    universe_group.add_argument(
        "--tickers", type=str, help="Comma-separated list of tickers (e.g. KO,PEP,MNST)"
    )
    parser.add_argument("--p-val", type=float, default=0.05, dest="p_val")
    parser.add_argument("--max-half-life", type=float, default=45.0, dest="max_half_life")
    parser.add_argument("--min-half-life", type=float, default=1.0, dest="min_half_life")
    parser.add_argument("--lookback-years", type=float, default=3.0, dest="lookback_years")
    return parser.parse_args(argv)


def _resolve_universe(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.sector:
        key = SECTOR_ALIASES.get(args.sector.lower())
        if key is None:
            valid = ", ".join(sorted(SECTOR_ALIASES))
            raise SystemExit(f"Unknown sector '{args.sector}'. Choices: {valid}")
        return SECTOR_UNIVERSES[key], key
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        raise SystemExit("--tickers must contain at least 2 comma-separated symbols")
    return tickers, "CUSTOM"


def _signal_tag(p_value: float) -> str:
    if p_value <= 0.01:
        return "*** STRONG"
    if p_value <= 0.03:
        return "**  MODERATE"
    return "*   WEAK"


def _print_results_table(results: pd.DataFrame) -> None:
    if results.empty:
        print("\nNo pairs cleared the cointegration and half-life filters.")
        return

    header = (
        f"{'Pair':<16}{'p-value':>10}{'t-stat':>10}{'Half-life':>12}"
        f"{'Hedge Ratio':>13}{'Z-score':>10}   {'Signal'}"
    )
    print()
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        pair = f"{row['Ticker_A']}/{row['Ticker_B']}"
        print(
            f"{pair:<16}"
            f"{row['p_value']:>10.4f}"
            f"{row['t_stat']:>10.3f}"
            f"{row['half_life_days']:>12.1f}"
            f"{row['hedge_ratio']:>13.4f}"
            f"{row['current_zscore']:>10.2f}   "
            f"{_signal_tag(row['p_value'])}"
        )
    print("-" * len(header))
    print(f"{len(results)} qualifying pair(s) found.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    tickers, sector_label = _resolve_universe(args)

    end_date = date.today()
    start_date = end_date - timedelta(days=int(365 * args.lookback_years))

    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    print(f"Universe: {sector_label} ({len(tickers)} tickers: {', '.join(tickers)})")
    print(f"Period:   {start_date} to {end_date}")
    print(f"Filters:  p-value <= {args.p_val}, half-life in [{args.min_half_life}, {args.max_half_life}] days")
    print(f"\nScanning {n_pairs} pairs in {sector_label.title().replace('_', ' ')}...")

    screener = PairScreener(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        p_value_threshold=args.p_val,
        max_half_life=args.max_half_life,
        min_half_life=args.min_half_life,
    )

    print("Fetching and validating universe data...")
    universe = screener.fetch_universe_data()
    print(f"  {universe.shape[1]} of {len(tickers)} tickers passed data-quality checks "
          f"({universe.shape[0]} aligned trading days).")

    is_tty = sys.stdout.isatty()

    def _progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            if is_tty:
                print(f"\r  Testing pair {done}/{total}...", end="", flush=True)
            else:
                print(f"  Testing pair {done}/{total}...")

    results = screener.screen_all_pairs(progress_callback=_progress)
    print()

    _print_results_table(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"screened_pairs_{sector_label.lower()}_{timestamp}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
