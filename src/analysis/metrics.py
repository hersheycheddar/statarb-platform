"""Performance metrics for backtested pairs-trading strategies.

Statistical rationale: raw P&L is not comparable across time periods or
strategies without normalizing for volatility and drawdown. These
metrics translate an equity curve and trade log into the standard
risk-adjusted return statistics used to judge whether a mean-reversion
edge is real and tradable, per SPEC.md section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.04


def total_return(equity: pd.Series) -> float:
    """Total return over the full equity curve, as a percentage."""
    if len(equity) < 2:
        raise ValueError("need at least 2 equity observations")
    return float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)


def annualized_return(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate (CAGR), as a percentage.

    Statistical rationale: CAGR expresses the total return as an
    equivalent constant annual rate, making strategies run over different
    lengths of time comparable.
    """
    if len(equity) < 2:
        raise ValueError("need at least 2 equity observations")
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        raise ValueError("insufficient time span to annualize")
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -100.0
    cagr = growth ** (1.0 / years) - 1.0
    return float(cagr * 100.0)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of a periodic returns series.

    Statistical rationale: Sharpe measures excess return per unit of
    volatility, so a strategy can be judged on risk-adjusted rather than
    raw performance. The annual risk-free rate is converted to the same
    period as `returns` before being subtracted.
    """
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        raise ValueError("need at least 2 return observations")
    period_rf = risk_free_rate / periods_per_year
    excess_returns = clean_returns - period_rf
    std = excess_returns.std()
    if std == 0:
        return 0.0
    return float(excess_returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline of the equity curve, as a (negative) percentage."""
    if len(equity) < 1:
        raise ValueError("need at least 1 equity observation")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100.0)


def calmar_ratio(annualized_return_pct: float, max_drawdown_pct: float) -> float:
    """Annualized return divided by the magnitude of max drawdown.

    Statistical rationale: Calmar penalizes strategies whose returns come
    with deep drawdowns, which are harder to sustain operationally and
    psychologically than volatility alone captures.
    """
    if max_drawdown_pct == 0:
        return float("inf") if annualized_return_pct > 0 else 0.0
    return float(annualized_return_pct / abs(max_drawdown_pct))


def win_rate(trade_log: pd.DataFrame) -> float:
    """Percentage of closed trades with positive net P&L."""
    if trade_log.empty:
        return 0.0
    wins = (trade_log["net_pnl"] > 0).sum()
    return float(wins / len(trade_log) * 100.0)


def profit_factor(trade_log: pd.DataFrame) -> float:
    """Gross profit divided by gross loss across all closed trades.

    Statistical rationale: unlike win rate, profit factor accounts for
    the size of wins vs. losses -- a strategy can win most trades and
    still lose money overall if its losers are large enough.
    """
    if trade_log.empty:
        return 0.0
    gross_profit = trade_log.loc[trade_log["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = trade_log.loc[trade_log["net_pnl"] < 0, "net_pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / abs(gross_loss))


def calculate_metrics(
    portfolio: pd.DataFrame,
    trade_log: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the full performance metrics dict from a backtest's outputs.

    `portfolio` must contain `equity` and `returns` columns (see
    src/backtest/engine.py's PairsBacktester output); `trade_log` must
    contain a `net_pnl` column.
    """
    equity = portfolio["equity"]
    returns = portfolio["returns"]

    annual_ret = annualized_return(equity, periods_per_year)
    max_dd = max_drawdown(equity)

    return {
        "total_return_pct": total_return(equity),
        "annualized_return_pct": annual_ret,
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar_ratio(annual_ret, max_dd),
        "win_rate_pct": win_rate(trade_log),
        "profit_factor": profit_factor(trade_log),
        "num_trades": int(len(trade_log)),
    }


def format_metrics_summary(metrics: dict[str, float]) -> str:
    """Render a metrics dict as an aligned, fixed-width console summary block."""

    def fmt(value: float) -> str:
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return f"{value:,.2f}"

    width = 40
    lines = [
        "-" * width,
        "PERFORMANCE SUMMARY",
        "-" * width,
        f"{'Total Return:':<24}{fmt(metrics['total_return_pct']):>10} %",
        f"{'Annualized Return (CAGR):':<24}{fmt(metrics['annualized_return_pct']):>10} %",
        f"{'Sharpe Ratio:':<24}{fmt(metrics['sharpe_ratio']):>10}",
        f"{'Max Drawdown:':<24}{fmt(metrics['max_drawdown_pct']):>10} %",
        f"{'Calmar Ratio:':<24}{fmt(metrics['calmar_ratio']):>10}",
        f"{'Win Rate:':<24}{fmt(metrics['win_rate_pct']):>10} %",
        f"{'Profit Factor:':<24}{fmt(metrics['profit_factor']):>10}",
        f"{'Number of Trades:':<24}{metrics['num_trades']:>10}",
        "-" * width,
    ]
    return "\n".join(lines)
