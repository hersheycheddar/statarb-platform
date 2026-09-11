"""Unit tests for src.data.loader.MarketDataLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import MarketDataLoader, RAW_COLUMNS


def _make_raw(dates: pd.DatetimeIndex, base_price: float) -> pd.DataFrame:
    """Build a synthetic raw OHLCV frame for a fake ticker."""
    n = len(dates)
    prices = [base_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


class TestCacheRoundtrip:
    def test_writes_parquet_cache_on_first_fetch(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        raw = _make_raw(dates, 100.0)

        calls = {"count": 0}

        def fake_fetch(self, ticker, start, end):
            calls["count"] += 1
            return raw

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader = MarketDataLoader(
            ["AAA"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        panel = loader.fetch()

        cache_file = cache_dir / "AAA_1d.parquet"
        assert cache_file.exists()
        assert calls["count"] == 1
        assert not panel.empty

    def test_second_fetch_reads_from_cache_not_api(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        raw = _make_raw(dates, 100.0)

        calls = {"count": 0}

        def fake_fetch(self, ticker, start, end):
            calls["count"] += 1
            return raw

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader1 = MarketDataLoader(
            ["AAA"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        loader1.fetch()
        assert calls["count"] == 1

        # Second loader over the same (fully covered) range must not call the API.
        def failing_fetch(self, ticker, start, end):
            raise AssertionError("API should not be called when cache covers the range")

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", failing_fetch)

        loader2 = MarketDataLoader(
            ["AAA"], start="2024-01-03", end="2024-01-09", cache_dir=cache_dir
        )
        panel = loader2.fetch()
        assert not panel.empty
        assert panel.index.min() >= pd.Timestamp("2024-01-03", tz="UTC")
        assert panel.index.max() <= pd.Timestamp("2024-01-09", tz="UTC")

    def test_roundtrip_preserves_values(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-02-01", "2024-02-08", freq="B", tz="UTC")
        raw = _make_raw(dates, 50.0)

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", lambda self, t, s, e: raw)

        loader1 = MarketDataLoader(
            ["BBB"], start="2024-02-01", end="2024-02-08", cache_dir=cache_dir
        )
        panel1 = loader1.fetch()

        monkeypatch.setattr(
            MarketDataLoader,
            "_fetch_from_api",
            lambda self, t, s, e: (_ for _ in ()).throw(AssertionError("should not hit API")),
        )
        loader2 = MarketDataLoader(
            ["BBB"], start="2024-02-01", end="2024-02-08", cache_dir=cache_dir
        )
        panel2 = loader2.fetch()

        pd.testing.assert_series_equal(panel1["BBB"], panel2["BBB"])

    def test_cache_extended_when_range_not_covered(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first_dates = pd.date_range("2024-03-01", "2024-03-08", freq="B", tz="UTC")
        monkeypatch.setattr(
            MarketDataLoader, "_fetch_from_api", lambda self, t, s, e: _make_raw(first_dates, 10.0)
        )
        loader1 = MarketDataLoader(
            ["CCC"], start="2024-03-01", end="2024-03-08", cache_dir=cache_dir
        )
        loader1.fetch()

        second_dates = pd.date_range("2024-03-01", "2024-03-20", freq="B", tz="UTC")
        calls = {"count": 0}

        def fake_fetch(self, ticker, start, end):
            calls["count"] += 1
            return _make_raw(second_dates, 10.0)

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader2 = MarketDataLoader(
            ["CCC"], start="2024-03-01", end="2024-03-20", cache_dir=cache_dir
        )
        panel = loader2.fetch()

        assert calls["count"] == 1
        assert panel.index.max() >= pd.Timestamp("2024-03-19", tz="UTC")


class TestMissingDataHandling:
    def test_short_gap_is_forward_filled(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        full_dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        raw_a = _make_raw(full_dates, 100.0)

        # Ticker B is missing a single day in the middle (a 1-day gap).
        gapped_dates = full_dates.delete(3)
        raw_b = _make_raw(full_dates, 200.0).loc[gapped_dates]

        def fake_fetch(self, ticker, start, end):
            return raw_a if ticker == "AAA" else raw_b

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader = MarketDataLoader(
            ["AAA", "BBB"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        panel = loader.fetch()

        # The gap in BBB should have been forward-filled, so no rows dropped.
        assert len(panel) == len(full_dates)
        assert panel["BBB"].isna().sum() == 0

    def test_non_overlapping_series_yields_near_empty_panel(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates_a = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        dates_b = pd.date_range("2024-06-01", "2024-06-10", freq="B", tz="UTC")

        def fake_fetch(self, ticker, start, end):
            return _make_raw(dates_a, 100.0) if ticker == "AAA" else _make_raw(dates_b, 200.0)

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader = MarketDataLoader(
            ["AAA", "BBB"], start="2024-01-02", end="2024-06-10", cache_dir=cache_dir
        )
        panel = loader.fetch()

        # The 2-day forward-fill limit may bridge at most the first couple of
        # rows at the start of the second block; there is no genuine overlap
        # so nothing deeper into either block should survive.
        assert len(panel) <= 2
        assert pd.Timestamp("2024-01-05", tz="UTC") not in panel.index
        assert pd.Timestamp("2024-06-07", tz="UTC") not in panel.index

    def test_long_gap_beyond_ffill_limit_is_dropped(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        full_dates = pd.date_range("2024-01-02", "2024-01-16", freq="B", tz="UTC")
        raw_a = _make_raw(full_dates, 100.0)

        # Ticker B has a 4-business-day gap in the middle (exceeds the 2-day ffill limit).
        gapped_dates = full_dates.delete([4, 5, 6, 7])
        raw_b = _make_raw(full_dates, 200.0).loc[gapped_dates]

        def fake_fetch(self, ticker, start, end):
            return raw_a if ticker == "AAA" else raw_b

        monkeypatch.setattr(MarketDataLoader, "_fetch_from_api", fake_fetch)

        loader = MarketDataLoader(
            ["AAA", "BBB"], start="2024-01-02", end="2024-01-16", cache_dir=cache_dir
        )
        panel = loader.fetch()

        assert len(panel) < len(full_dates)
        assert panel.isna().sum().sum() == 0


class TestIndexingAndColumns:
    def test_index_is_named_date_datetimeindex(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        monkeypatch.setattr(
            MarketDataLoader, "_fetch_from_api", lambda self, t, s, e: _make_raw(dates, 100.0)
        )

        loader = MarketDataLoader(
            ["AAA"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        panel = loader.fetch()

        assert isinstance(panel.index, pd.DatetimeIndex)
        assert panel.index.name == "date"
        assert panel.index.is_monotonic_increasing

    def test_columns_match_requested_tickers_in_order(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        monkeypatch.setattr(
            MarketDataLoader, "_fetch_from_api", lambda self, t, s, e: _make_raw(dates, 100.0)
        )

        loader = MarketDataLoader(
            ["ZZZ", "AAA", "MMM"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        panel = loader.fetch()

        assert list(panel.columns) == ["ZZZ", "AAA", "MMM"]
        assert panel.columns.name == "ticker"

    def test_duplicate_tickers_are_deduplicated(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="UTC")
        monkeypatch.setattr(
            MarketDataLoader, "_fetch_from_api", lambda self, t, s, e: _make_raw(dates, 100.0)
        )

        loader = MarketDataLoader(
            ["AAA", "AAA"], start="2024-01-02", end="2024-01-10", cache_dir=cache_dir
        )
        assert loader.tickers == ["AAA"]


class TestValidation:
    def test_empty_ticker_list_raises(self, cache_dir: Path) -> None:
        with pytest.raises(ValueError):
            MarketDataLoader([], start="2024-01-01", end="2024-01-10", cache_dir=cache_dir)

    def test_start_after_end_raises(self, cache_dir: Path) -> None:
        with pytest.raises(ValueError):
            MarketDataLoader(
                ["AAA"], start="2024-01-10", end="2024-01-01", cache_dir=cache_dir
            )

    def test_cache_dir_is_created(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nested" / "cache"
        assert not cache_dir.exists()
        MarketDataLoader(["AAA"], start="2024-01-01", end="2024-01-10", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_raw_columns_constant_shape(self) -> None:
        assert RAW_COLUMNS == ["open", "high", "low", "close", "adj_close", "volume"]
