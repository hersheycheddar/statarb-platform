"""Unit tests for src.analysis.metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.metrics import (
    calculate_metrics,
    calmar_ratio,
    format_metrics_summary,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)


def _date_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")


class TestTotalReturn:
    def test_simple_growth(self) -> None:
        equity = pd.Series([100.0, 110.0, 121.0], index=_date_index(3))
        assert total_return(equity) == pytest.approx(21.0)


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0, 130.0], index=_date_index(4))
        # Peak 120 -> trough 90 => -25%
        assert max_drawdown(equity) == pytest.approx(-25.0)

    def test_monotonic_increase_has_zero_drawdown(self) -> None:
        equity = pd.Series([100.0, 105.0, 110.0], index=_date_index(3))
        assert max_drawdown(equity) == pytest.approx(0.0)


class TestSharpeRatio:
    def test_zero_volatility_returns_zero(self) -> None:
        returns = pd.Series([0.001] * 10, index=_date_index(10))
        assert sharpe_ratio(returns) == pytest.approx(0.0)

    def test_positive_excess_returns_give_positive_sharpe(self) -> None:
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01, 0.02], index=_date_index(6))
        assert sharpe_ratio(returns, risk_free_rate=0.0) > 0


class TestCalmarRatio:
    def test_basic_ratio(self) -> None:
        assert calmar_ratio(20.0, -10.0) == pytest.approx(2.0)

    def test_zero_drawdown_with_positive_return_is_inf(self) -> None:
        assert calmar_ratio(15.0, 0.0) == float("inf")


class TestWinRateAndProfitFactor:
    def test_win_rate_and_profit_factor(self) -> None:
        trade_log = pd.DataFrame({"net_pnl": [100.0, -50.0, 200.0, -25.0]})
        assert win_rate(trade_log) == pytest.approx(50.0)
        assert profit_factor(trade_log) == pytest.approx(300.0 / 75.0)

    def test_empty_trade_log(self) -> None:
        trade_log = pd.DataFrame(columns=["net_pnl"])
        assert win_rate(trade_log) == 0.0
        assert profit_factor(trade_log) == 0.0

    def test_no_losses_gives_inf_profit_factor(self) -> None:
        trade_log = pd.DataFrame({"net_pnl": [100.0, 50.0]})
        assert profit_factor(trade_log) == float("inf")


class TestCalculateMetrics:
    def test_full_pipeline_keys_present(self) -> None:
        idx = _date_index(5)
        portfolio = pd.DataFrame(
            {
                "equity": [100_000, 101_000, 99_500, 102_000, 103_000],
                "returns": [0.0, 0.01, -0.0149, 0.0251, 0.0098],
            },
            index=idx,
        )
        trade_log = pd.DataFrame(
            {"net_pnl": [500.0, -200.0]},
            index=[0, 1],
        )

        metrics = calculate_metrics(portfolio, trade_log)

        expected_keys = {
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "calmar_ratio",
            "win_rate_pct",
            "profit_factor",
            "num_trades",
        }
        assert set(metrics) == expected_keys
        assert metrics["num_trades"] == 2

    def test_format_metrics_summary_contains_all_labels(self) -> None:
        metrics = {
            "total_return_pct": 12.5,
            "annualized_return_pct": 8.0,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -5.5,
            "calmar_ratio": 1.45,
            "win_rate_pct": 60.0,
            "profit_factor": 2.1,
            "num_trades": 10,
        }
        summary = format_metrics_summary(metrics)
        for label in [
            "Total Return",
            "Annualized Return",
            "Sharpe Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
            "Profit Factor",
            "Number of Trades",
        ]:
            assert label in summary
