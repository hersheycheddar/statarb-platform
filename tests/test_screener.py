"""Unit tests for src.engine.screener.PairScreener."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import MarketDataLoader
from src.engine.pairs import CointegrationResult, PairsAnalyzer
from src.engine.screener import SECTOR_UNIVERSES, PairScreener


def _date_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")


def _make_raw(dates: pd.DatetimeIndex, base_price: float) -> pd.DataFrame:
    n = len(dates)
    prices = [base_price + i * 0.01 for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


def _fake_coint_result(p_value: float, t_stat: float = -4.0) -> CointegrationResult:
    return CointegrationResult(
        t_statistic=t_stat,
        p_value=p_value,
        critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
    )


class TestUniverseDataAlignment:
    def test_two_full_history_tickers_align(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.bdate_range("2024-01-02", "2024-06-28", tz="UTC")
        raw = {"AAA": _make_raw(dates, 100.0), "BBB": _make_raw(dates, 50.0)}

        monkeypatch.setattr(
            MarketDataLoader, "_fetch_from_api", lambda self, ticker, s, e: raw[ticker]
        )

        screener = PairScreener(
            ["AAA", "BBB"], start_date="2024-01-02", end_date="2024-06-28", cache_dir=tmp_path
        )
        universe = screener.fetch_universe_data()

        assert list(universe.columns) == ["AAA", "BBB"]
        assert isinstance(universe.index, pd.DatetimeIndex)
        assert universe.isna().sum().sum() == 0
        assert len(universe) > 0

    def test_ticker_with_excessive_missing_data_is_dropped(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dates = pd.bdate_range("2024-01-02", "2024-06-28", tz="UTC")
        good = _make_raw(dates, 100.0)
        sparse_dates = dates[: max(1, len(dates) // 3)]
        bad = _make_raw(dates, 50.0).loc[sparse_dates]

        def fake_fetch(self, ticker, start, end):
            return good if ticker == "AAA" else bad

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        screener = PairScreener(
            ["AAA", "BBB"], start_date="2024-01-02", end_date="2024-06-28", cache_dir=tmp_path
        )
        with pytest.warns(UserWarning, match="BBB"):
            universe = screener.fetch_universe_data()

        assert list(universe.columns) == ["AAA"]

    def test_failed_download_is_skipped_with_warning(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dates = pd.bdate_range("2024-01-02", "2024-06-28", tz="UTC")
        good = _make_raw(dates, 100.0)

        def fake_fetch(self, ticker, start, end):
            if ticker == "BAD":
                raise RuntimeError("simulated network failure")
            return good

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        screener = PairScreener(
            ["AAA", "BAD"], start_date="2024-01-02", end_date="2024-06-28", cache_dir=tmp_path
        )
        with pytest.warns(UserWarning, match="BAD"):
            universe = screener.fetch_universe_data()

        assert list(universe.columns) == ["AAA"]

    def test_all_tickers_failing_raises(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        def always_fail(self, ticker, start, end):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", always_fail)

        screener = PairScreener(
            ["AAA", "BBB"], start_date="2024-01-02", end_date="2024-06-28", cache_dir=tmp_path
        )
        with pytest.raises(ValueError):
            screener.fetch_universe_data()


class TestScreeningLogic:
    def test_pair_within_thresholds_is_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 100
        idx = _date_index(n)
        theta = 0.15  # half-life = ln(2)/0.15 ~= 4.62 days
        spread_shape = [10.0 * (1 - theta) ** t for t in range(n)]
        universe = pd.DataFrame(
            {"AAA": spread_shape, "BBB": [0.0] * n}, index=idx
        )

        screener = PairScreener(["AAA", "BBB"], start_date="2024-01-01", end_date="2024-06-01")
        screener.universe_data = universe

        monkeypatch.setattr(PairScreener, "_full_sample_hedge_ratio", lambda self, a, b: 1.0)
        monkeypatch.setattr(
            PairsAnalyzer, "test_cointegration", lambda self, a, b: _fake_coint_result(0.01)
        )

        results = screener.screen_all_pairs()

        assert len(results) == 1
        row = results.iloc[0]
        assert (row["Ticker_A"], row["Ticker_B"]) == ("AAA", "BBB")
        assert row["half_life_days"] == pytest.approx(np.log(2) / theta, rel=1e-6)

    def test_pair_exceeding_p_value_threshold_is_excluded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n = 100
        idx = _date_index(n)
        theta = 0.15
        spread_shape = [10.0 * (1 - theta) ** t for t in range(n)]
        universe = pd.DataFrame(
            {"AAA": spread_shape, "BBB": [0.0] * n}, index=idx
        )

        screener = PairScreener(
            ["AAA", "BBB"],
            start_date="2024-01-01",
            end_date="2024-06-01",
            p_value_threshold=0.05,
        )
        screener.universe_data = universe

        monkeypatch.setattr(PairScreener, "_full_sample_hedge_ratio", lambda self, a, b: 1.0)
        # p-value above the threshold despite an otherwise-qualifying half-life.
        monkeypatch.setattr(
            PairsAnalyzer, "test_cointegration", lambda self, a, b: _fake_coint_result(0.20)
        )

        results = screener.screen_all_pairs()

        assert results.empty

    def test_pair_exceeding_max_half_life_is_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 100
        idx = _date_index(n)
        theta = 0.001  # half-life ~= 693 days, far beyond any reasonable max
        spread_shape = [10.0 * (1 - theta) ** t for t in range(n)]
        universe = pd.DataFrame(
            {"AAA": spread_shape, "BBB": [0.0] * n}, index=idx
        )

        screener = PairScreener(
            ["AAA", "BBB"],
            start_date="2024-01-01",
            end_date="2024-06-01",
            p_value_threshold=0.05,
            max_half_life=45.0,
        )
        screener.universe_data = universe

        monkeypatch.setattr(PairScreener, "_full_sample_hedge_ratio", lambda self, a, b: 1.0)
        monkeypatch.setattr(
            PairsAnalyzer, "test_cointegration", lambda self, a, b: _fake_coint_result(0.01)
        )

        results = screener.screen_all_pairs()

        assert results.empty

    def test_pair_below_min_half_life_is_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 100
        idx = _date_index(n)
        theta = 0.99  # half-life ~= 0.7 days, below the default min of 1.0
        spread_shape = [10.0 * (1 - theta) ** t for t in range(n)]
        universe = pd.DataFrame(
            {"AAA": spread_shape, "BBB": [0.0] * n}, index=idx
        )

        screener = PairScreener(
            ["AAA", "BBB"], start_date="2024-01-01", end_date="2024-06-01", min_half_life=1.0
        )
        screener.universe_data = universe

        monkeypatch.setattr(PairScreener, "_full_sample_hedge_ratio", lambda self, a, b: 1.0)
        monkeypatch.setattr(
            PairsAnalyzer, "test_cointegration", lambda self, a, b: _fake_coint_result(0.01)
        )

        results = screener.screen_all_pairs()

        assert results.empty

    def test_results_sorted_by_p_value_then_half_life(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 100
        idx = _date_index(n)

        def _decay(theta: float) -> list[float]:
            return [10.0 * (1 - theta) ** t for t in range(n)]

        universe = pd.DataFrame(
            {
                "AAA": _decay(0.20),  # half-life ~= 3.47 days
                "BBB": _decay(0.10),  # half-life ~= 6.93 days
                "CCC": [0.0] * n,
            },
            index=idx,
        )

        screener = PairScreener(
            ["AAA", "BBB", "CCC"], start_date="2024-01-01", end_date="2024-06-01"
        )
        screener.universe_data = universe

        monkeypatch.setattr(PairScreener, "_full_sample_hedge_ratio", lambda self, a, b: 1.0)

        def fake_coint(self, a, b):
            names = {a.name, b.name}
            if names == {"AAA", "CCC"}:
                return _fake_coint_result(0.01)
            if names == {"BBB", "CCC"}:
                return _fake_coint_result(0.001)
            return _fake_coint_result(0.5)  # AAA/BBB: excluded

        monkeypatch.setattr(PairsAnalyzer, "test_cointegration", fake_coint)

        results = screener.screen_all_pairs()

        assert list(zip(results["Ticker_A"], results["Ticker_B"])) == [
            ("BBB", "CCC"),
            ("AAA", "CCC"),
        ]


class TestValidation:
    def test_requires_at_least_two_tickers(self) -> None:
        with pytest.raises(ValueError):
            PairScreener(["AAA"], start_date="2024-01-01", end_date="2024-06-01")

    def test_min_half_life_must_be_less_than_max(self) -> None:
        with pytest.raises(ValueError):
            PairScreener(
                ["AAA", "BBB"],
                start_date="2024-01-01",
                end_date="2024-06-01",
                min_half_life=50.0,
                max_half_life=10.0,
            )

    def test_start_before_end(self) -> None:
        with pytest.raises(ValueError):
            PairScreener(["AAA", "BBB"], start_date="2024-06-01", end_date="2024-01-01")


class TestSectorUniverses:
    def test_expected_sectors_present(self) -> None:
        assert set(SECTOR_UNIVERSES) == {
            "ENERGY",
            "FINANCIALS",
            "COMMODITY_ETFS",
            "TECH_PAYMENTS",
            "DUAL_CLASS",
        }

    def test_energy_has_eight_tickers(self) -> None:
        assert len(SECTOR_UNIVERSES["ENERGY"]) == 8

    def test_no_duplicate_tickers_within_a_sector(self) -> None:
        for sector, tickers in SECTOR_UNIVERSES.items():
            assert len(tickers) == len(set(tickers)), f"{sector} has duplicate tickers"
