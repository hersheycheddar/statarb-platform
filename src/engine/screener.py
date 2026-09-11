"""Sector-wide cointegration screening for pairs-trading candidate discovery.

Statistical rationale: hand-picking pairs from intuition (e.g. "Coke and
Pepsi should move together") introduces selection bias and misses less
obvious relationships. Screening every combination within a sector against
a consistent statistical bar -- Engle-Granger p-value and OU half-life --
surfaces only pairs with evidence of a genuine, tradable mean-reverting
relationship.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.data.loader import DEFAULT_CACHE_DIR, MarketDataLoader
from src.engine.pairs import PairsAnalyzer

SECTOR_UNIVERSES: dict[str, list[str]] = {
    "ENERGY": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO"],
    "FINANCIALS": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "COMMODITY_ETFS": ["GLD", "IAU", "SLV", "SIVR", "USO", "BNO", "CPER"],
    "TECH_PAYMENTS": ["V", "MA", "PYPL", "FIS", "FISV"],
    "DUAL_CLASS": ["GOOG", "GOOGL", "FOX", "FOXA", "NWS", "NWSA"],
}

RESULT_COLUMNS = [
    "Ticker_A",
    "Ticker_B",
    "p_value",
    "t_stat",
    "half_life_days",
    "hedge_ratio",
    "current_zscore",
]

MAX_MISSING_PCT = 0.05


class PairScreener:
    """Scans a universe of tickers for statistically valid cointegrated pairs."""

    def __init__(
        self,
        tickers: list[str],
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        p_value_threshold: float = 0.05,
        max_half_life: float = 45.0,
        min_half_life: float = 1.0,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        if len(tickers) < 2:
            raise ValueError("need at least 2 tickers to screen pairs")
        if min_half_life >= max_half_life:
            raise ValueError("min_half_life must be less than max_half_life")
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            raise ValueError("start_date must be before end_date")

        self.tickers = list(dict.fromkeys(tickers))
        self.start_date = start_date
        self.end_date = end_date
        self.p_value_threshold = p_value_threshold
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.cache_dir = cache_dir
        self.universe_data: pd.DataFrame | None = None
        self._analyzer = PairsAnalyzer()

    def fetch_universe_data(self) -> pd.DataFrame:
        """Batch-download and clean the full price panel for `self.tickers`.

        Statistical rationale: cointegration testing needs long, gap-free
        overlapping histories. A ticker that fails to download, or is
        missing more than 5% of the requested trading calendar, would
        silently shrink or bias the inner join for every pair it appears
        in -- so it is dropped from the universe up front instead.
        """
        expected_days = len(pd.bdate_range(self.start_date, self.end_date))
        clean_series: dict[str, pd.Series] = {}

        for ticker in self.tickers:
            try:
                loader = MarketDataLoader(
                    [ticker],
                    start=self.start_date,
                    end=self.end_date,
                    cache_dir=self.cache_dir,
                )
                panel = loader.fetch()
            except Exception as exc:  # a single bad ticker must not abort the whole screen
                warnings.warn(f"Skipping {ticker}: failed to download ({exc})", stacklevel=2)
                continue

            series = panel.get(ticker, pd.Series(dtype="float64"))
            missing_pct = 1.0 - (len(series) / expected_days) if expected_days else 1.0
            if missing_pct > MAX_MISSING_PCT:
                warnings.warn(
                    f"Skipping {ticker}: {missing_pct:.1%} missing data exceeds "
                    f"{MAX_MISSING_PCT:.0%} threshold",
                    stacklevel=2,
                )
                continue

            clean_series[ticker] = series

        if not clean_series:
            raise ValueError("no tickers survived data availability/quality filtering")

        universe = pd.DataFrame(clean_series).sort_index().dropna(how="any")
        universe.columns.name = "ticker"
        self.universe_data = universe
        return universe

    def _full_sample_hedge_ratio(self, series_a: pd.Series, series_b: pd.Series) -> float:
        """Static OLS hedge ratio over the pair's full overlapping history.

        Statistical rationale: this characterizes a candidate pair for the
        screener's report (spread definition, current z-score) only -- it
        is not used to generate trade signals, so unlike
        `PairsAnalyzer.calculate_hedge_ratio` it needs no lookahead lag.
        """
        covariance = series_a.cov(series_b)
        variance = series_b.var()
        return float(covariance / variance)

    def _current_zscore(self, spread: pd.Series) -> float:
        """Z-score of the most recent spread observation vs. its full-sample distribution."""
        mean = spread.mean()
        std = spread.std()
        if std == 0:
            return 0.0
        return float((spread.iloc[-1] - mean) / std)

    def screen_all_pairs(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> pd.DataFrame:
        """Test every unique ticker pair for tradable cointegration.

        Statistical rationale: screening the full N*(N-1)/2 combinations
        within a universe, rather than hand-picking pairs, avoids the
        selection bias of only testing relationships an analyst already
        suspects, and applies a uniform statistical bar to all of them.
        """
        if self.universe_data is None:
            self.fetch_universe_data()
        assert self.universe_data is not None

        pairs = list(itertools.combinations(self.universe_data.columns, 2))
        total = len(pairs)
        results: list[dict[str, object]] = []

        for i, (ticker_a, ticker_b) in enumerate(pairs, start=1):
            series_a = self.universe_data[ticker_a]
            series_b = self.universe_data[ticker_b]

            try:
                coint_result = self._analyzer.test_cointegration(series_a, series_b)
            except ValueError:
                if progress_callback:
                    progress_callback(i, total)
                continue

            if coint_result.p_value <= self.p_value_threshold:
                hedge_ratio = self._full_sample_hedge_ratio(series_a, series_b)
                spread = series_a - hedge_ratio * series_b

                try:
                    half_life = self._analyzer.calculate_half_life(spread)
                except ValueError:
                    half_life = None

                if half_life is not None and self.min_half_life <= half_life <= self.max_half_life:
                    results.append(
                        {
                            "Ticker_A": ticker_a,
                            "Ticker_B": ticker_b,
                            "p_value": coint_result.p_value,
                            "t_stat": coint_result.t_statistic,
                            "half_life_days": half_life,
                            "hedge_ratio": hedge_ratio,
                            "current_zscore": self._current_zscore(spread),
                        }
                    )

            if progress_callback:
                progress_callback(i, total)

        results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
        results_df = results_df.sort_values(
            ["p_value", "half_life_days"], ascending=[True, True]
        ).reset_index(drop=True)
        return results_df
