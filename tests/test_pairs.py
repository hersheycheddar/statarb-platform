"""Unit tests for src.engine.pairs.PairsAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.pairs import CointegrationResult, PairsAnalyzer


@pytest.fixture
def analyzer() -> PairsAnalyzer:
    return PairsAnalyzer()


def _date_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")


class TestCointegration:
    def test_cointegrated_series_have_low_p_value(self, analyzer: PairsAnalyzer) -> None:
        rng = np.random.default_rng(42)
        n = 500
        common_walk = np.cumsum(rng.normal(0, 1, n))
        noise = rng.normal(0, 0.5, n)  # stationary noise -> A - 1.5*B is stationary
        series_b = pd.Series(common_walk + 100, index=_date_index(n))
        series_a = pd.Series(1.5 * series_b.values + noise, index=_date_index(n))

        result = analyzer.test_cointegration(series_a, series_b)

        assert isinstance(result, CointegrationResult)
        assert result.p_value < 0.05
        assert set(result.critical_values) == {"1%", "5%", "10%"}

    def test_independent_random_walks_are_not_cointegrated(self, analyzer: PairsAnalyzer) -> None:
        rng = np.random.default_rng(7)
        n = 500
        series_a = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_date_index(n))
        series_b = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_date_index(n))

        result = analyzer.test_cointegration(series_a, series_b)

        assert result.p_value > 0.05

    def test_raises_on_too_few_observations(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(2)
        series_a = pd.Series([1.0, 2.0], index=idx)
        series_b = pd.Series([1.0, 2.0], index=idx)
        with pytest.raises(ValueError):
            analyzer.test_cointegration(series_a, series_b)


class TestHedgeRatioNoLookahead:
    def test_warmup_period_is_nan(self, analyzer: PairsAnalyzer) -> None:
        rng = np.random.default_rng(1)
        n = 50
        window = 10
        series_b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=_date_index(n))
        series_a = pd.Series(2.0 * series_b.values + rng.normal(0, 0.1, n), index=_date_index(n))

        hedge_ratio = analyzer.calculate_hedge_ratio(series_a, series_b, window)

        # `window` rows are consumed by the rolling stat itself, plus 1 more for the shift.
        assert hedge_ratio.iloc[:window].isna().all()
        assert hedge_ratio.iloc[window:].notna().all()

    def test_shift_applied_beta_at_t_equals_raw_beta_at_t_minus_1(self, analyzer: PairsAnalyzer) -> None:
        rng = np.random.default_rng(2)
        n = 60
        window = 15
        series_b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50, index=_date_index(n))
        series_a = pd.Series(0.8 * series_b.values + rng.normal(0, 0.2, n), index=_date_index(n))

        raw_cov = series_a.rolling(window).cov(series_b)
        raw_var = series_b.rolling(window).var()
        raw_beta = raw_cov / raw_var

        hedge_ratio = analyzer.calculate_hedge_ratio(series_a, series_b, window)

        # hedge_ratio[t] must equal the *unshifted* rolling beta computed at t-1,
        # i.e. it must never incorporate information from bar t itself.
        np.testing.assert_allclose(
            hedge_ratio.iloc[window:].to_numpy(),
            raw_beta.iloc[window - 1 : -1].to_numpy(),
        )

    def test_estimated_beta_close_to_true_value(self, analyzer: PairsAnalyzer) -> None:
        rng = np.random.default_rng(3)
        n = 300
        window = 30
        true_beta = 1.75
        series_b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=_date_index(n))
        series_a = pd.Series(
            true_beta * series_b.values + rng.normal(0, 0.05, n), index=_date_index(n)
        )

        hedge_ratio = analyzer.calculate_hedge_ratio(series_a, series_b, window)

        assert hedge_ratio.dropna().mean() == pytest.approx(true_beta, abs=0.1)

    def test_raises_on_window_too_small(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(5)
        series_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        series_b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        with pytest.raises(ValueError):
            analyzer.calculate_hedge_ratio(series_a, series_b, 1)


class TestSpread:
    def test_spread_matches_formula(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(5)
        series_a = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
        series_b = pd.Series([5.0, 5.5, 6.0, 6.5, 7.0], index=idx)
        hedge_ratios = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0], index=idx)

        spread = analyzer.calculate_spread(series_a, series_b, hedge_ratios)

        expected = series_a - hedge_ratios * series_b
        pd.testing.assert_series_equal(spread, expected, check_names=False)

    def test_spread_propagates_hedge_ratio_nans(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(4)
        series_a = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        series_b = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
        hedge_ratios = pd.Series([np.nan, np.nan, 1.0, 1.0], index=idx)

        spread = analyzer.calculate_spread(series_a, series_b, hedge_ratios)

        assert spread.iloc[:2].isna().all()
        assert spread.iloc[2:].notna().all()


class TestZScore:
    def test_warmup_period_is_nan(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(30)
        spread = pd.Series(np.sin(np.linspace(0, 6, 30)), index=idx)
        window = 10

        zscore = analyzer.calculate_zscore(spread, window)

        assert zscore.iloc[: window - 1].isna().all()
        assert zscore.iloc[window - 1 :].notna().all()

    def test_zscore_formula(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(6)
        spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        window = 3

        zscore = analyzer.calculate_zscore(spread, window)

        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        expected = (spread - rolling_mean) / rolling_std
        pd.testing.assert_series_equal(zscore, expected, check_names=False)

    def test_raises_on_window_too_small(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(5)
        spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        with pytest.raises(ValueError):
            analyzer.calculate_zscore(spread, 1)


class TestHalfLife:
    def test_half_life_matches_known_ou_process(self, analyzer: PairsAnalyzer) -> None:
        theta = 0.1
        n = 200
        x0 = 10.0
        values = [x0 * (1 - theta) ** t for t in range(n)]
        spread = pd.Series(values, index=_date_index(n))

        half_life = analyzer.calculate_half_life(spread)
        expected = np.log(2) / theta

        assert half_life == pytest.approx(expected, rel=1e-6)

    def test_raises_on_insufficient_data(self, analyzer: PairsAnalyzer) -> None:
        idx = _date_index(2)
        spread = pd.Series([1.0, 2.0], index=idx)
        with pytest.raises(ValueError):
            analyzer.calculate_half_life(spread)

    def test_explosive_non_mean_reverting_series_returns_inf(self, analyzer: PairsAnalyzer) -> None:
        n = 30
        values = [1.0 * (1.1**t) for t in range(n)]
        spread = pd.Series(values, index=_date_index(n))

        half_life = analyzer.calculate_half_life(spread)

        assert half_life == float("inf")
