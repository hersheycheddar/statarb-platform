"""Unit tests for src.backtest.engine.PairsBacktester."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import PairsBacktester


def _date_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")


def _series(name: str, values: list[float], idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=idx, name=name)


class TestTransactionCosts:
    def test_round_trip_with_flat_prices_only_loses_fees(self) -> None:
        """With unmoving prices, a full open+close round trip's only P&L
        impact should be the transaction fees paid on each of the 4 fills."""
        idx = _date_index(6)
        price_a = _series("AAA", [100.0] * 6, idx)
        price_b = _series("BBB", [50.0] * 6, idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        # day0: flat. day1: z>entry_z -> SHORT opens on day2.
        # day2-3: hold. day3: z crosses back through exit_z -> closes on day4.
        zscore = _series("zscore", [0.0, 2.5, 2.5, 0.0, 0.0, 0.0], idx)

        backtester = PairsBacktester(initial_capital=100_000.0, transaction_fee_pct=0.0005)
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["gross_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert trade["fees"] > 0
        assert trade["net_pnl"] == pytest.approx(-trade["fees"])

        final_cash = portfolio["cash"].iloc[-1]
        assert final_cash == pytest.approx(100_000.0 - trade["fees"])
        assert final_cash < 100_000.0

    def test_higher_fee_pct_reduces_cash_more(self) -> None:
        idx = _date_index(6)
        price_a = _series("AAA", [100.0] * 6, idx)
        price_b = _series("BBB", [50.0] * 6, idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        zscore = _series("zscore", [0.0, 2.5, 2.5, 0.0, 0.0, 0.0], idx)

        cheap = PairsBacktester(transaction_fee_pct=0.0001)
        expensive = PairsBacktester(transaction_fee_pct=0.01)

        _, cheap_trades = cheap.run(price_a, price_b, hedge_ratio, spread, zscore)
        _, expensive_trades = expensive.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert expensive_trades.iloc[0]["fees"] > cheap_trades.iloc[0]["fees"]

    def test_zero_fee_round_trip_has_zero_net_pnl_on_flat_prices(self) -> None:
        idx = _date_index(6)
        price_a = _series("AAA", [100.0] * 6, idx)
        price_b = _series("BBB", [50.0] * 6, idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        zscore = _series("zscore", [0.0, 2.5, 2.5, 0.0, 0.0, 0.0], idx)

        backtester = PairsBacktester(transaction_fee_pct=0.0)
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert trades.iloc[0]["net_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert portfolio["cash"].iloc[-1] == pytest.approx(100_000.0)


class TestStopLoss:
    def test_stop_loss_exits_at_correct_threshold(self) -> None:
        """Position enters LONG the spread on a deep negative z-score, then
        the spread keeps diverging past -stop_loss_z (never reverting back
        toward 0) -- this must force an exit driven by the stop-loss branch,
        distinct from a normal mean-reversion exit."""
        idx = _date_index(6)
        price_a = _series("AAA", [100.0] * 6, idx)
        price_b = _series("BBB", [50.0] * 6, idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        # day1: z < -entry_z(-2.0) -> LONG opens day2.
        # day2: still beyond entry but inside stop-loss -> hold.
        # day3: z <= -stop_loss_z(-3.5) -> stop-loss triggers, closes day4.
        zscore = _series("zscore", [0.0, -2.5, -2.8, -3.6, -3.6, -3.6], idx)

        backtester = PairsBacktester(entry_z=2.0, exit_z=0.0, stop_loss_z=3.5)
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["type"] == "LONG"
        assert trade["entry_date"] == idx[2]
        assert trade["exit_date"] == idx[4]

        # Confirm the position was actually open through day3 and flat by day4.
        assert portfolio.loc[idx[2], "position"] == 1
        assert portfolio.loc[idx[3], "position"] == 1
        assert portfolio.loc[idx[4], "position"] == 0

    def test_no_stop_loss_when_z_stays_within_bounds(self) -> None:
        idx = _date_index(6)
        price_a = _series("AAA", [100.0] * 6, idx)
        price_b = _series("BBB", [50.0] * 6, idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        # Diverges but never past -stop_loss_z, and never reverts to exit_z either.
        zscore = _series("zscore", [0.0, -2.5, -2.8, -2.9, -3.0, -3.0], idx)

        backtester = PairsBacktester(entry_z=2.0, exit_z=0.0, stop_loss_z=3.5)
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert len(trades) == 0
        assert portfolio["position"].iloc[-1] == 1


class TestPositionFlattening:
    def test_position_and_holdings_are_zero_after_exit(self) -> None:
        idx = _date_index(6)
        price_a = _series("AAA", [100.0, 101.0, 99.0, 102.0, 98.0, 103.0], idx)
        price_b = _series("BBB", [50.0, 50.5, 49.5, 51.0, 49.0, 51.5], idx)
        hedge_ratio = _series("hedge_ratio", [1.0] * 6, idx)
        spread = price_a - hedge_ratio * price_b
        zscore = _series("zscore", [0.0, 2.5, 2.5, 0.0, 0.0, 0.0], idx)

        backtester = PairsBacktester()
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        exit_date = trades.iloc[0]["exit_date"]
        post_exit = portfolio.loc[exit_date:]

        assert (post_exit["position"] == 0).all()
        # With no open position, equity must exactly equal cash (no residual holdings).
        pd.testing.assert_series_equal(
            post_exit["equity"], post_exit["cash"], check_names=False
        )

    def test_flattening_holds_across_warmup_nans(self) -> None:
        idx = _date_index(5)
        price_a = _series("AAA", [100.0] * 5, idx)
        price_b = _series("BBB", [50.0] * 5, idx)
        hedge_ratio = _series("hedge_ratio", [float("nan"), float("nan"), 1.0, 1.0, 1.0], idx)
        spread = price_a - hedge_ratio.fillna(1.0) * price_b
        zscore = _series("zscore", [float("nan"), float("nan"), 0.0, 0.0, 0.0], idx)

        backtester = PairsBacktester()
        portfolio, trades = backtester.run(price_a, price_b, hedge_ratio, spread, zscore)

        assert len(trades) == 0
        assert (portfolio["position"] == 0).all()
        pd.testing.assert_series_equal(
            portfolio["equity"], portfolio["cash"], check_names=False
        )


class TestValidation:
    def test_stop_loss_must_exceed_entry_z(self) -> None:
        with pytest.raises(ValueError):
            PairsBacktester(entry_z=3.0, stop_loss_z=2.0)

    def test_negative_fee_raises(self) -> None:
        with pytest.raises(ValueError):
            PairsBacktester(transaction_fee_pct=-0.01)
