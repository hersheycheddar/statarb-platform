"""Portfolio simulation for a single-pair mean-reversion strategy.

Statistical rationale: given a spread and its rolling z-score from
`src/engine`, a mean-reversion strategy enters when the spread is an
extreme number of standard deviations from its trailing mean (expecting
reversion) and exits once it reverts toward the mean or, failing that,
once it diverges far enough to indicate the cointegrating relationship
has broken down (stop-loss). Per SPEC.md section 2's lookahead-bias
contract, a position decided from bar t's z-score is only ever filled at
bar t+1's price -- this module never trades on the same bar it observes
a signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TRADE_LOG_COLUMNS = [
    "entry_date",
    "exit_date",
    "pair",
    "type",
    "entry_price_a",
    "entry_price_b",
    "exit_price_a",
    "exit_price_b",
    "gross_pnl",
    "fees",
    "net_pnl",
]


@dataclass
class _OpenPosition:
    entry_date: pd.Timestamp
    side: int
    qty_a: float
    qty_b: float
    entry_price_a: float
    entry_price_b: float
    entry_fees: float


class PairsBacktester:
    """Simulates entering/exiting a single cointegrated pair from z-score signals.

    Statistical rationale: `entry_z` sets how extreme a deviation must be
    before betting on reversion (higher = fewer, higher-conviction
    trades); `exit_z` sets how much of the reversion to capture before
    closing; `stop_loss_z` bounds losses if the spread keeps diverging
    instead of reverting, which happens when the cointegrating
    relationship itself has broken down.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        stop_loss_z: float = 3.5,
        transaction_fee_pct: float = 0.0005,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if entry_z <= 0:
            raise ValueError("entry_z must be positive")
        if stop_loss_z <= entry_z:
            raise ValueError("stop_loss_z must be greater than entry_z")
        if transaction_fee_pct < 0:
            raise ValueError("transaction_fee_pct must be non-negative")

        self.initial_capital = initial_capital
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.transaction_fee_pct = transaction_fee_pct

    def _next_target(self, position_side: int, z_t: float) -> int:
        """Decide the position to hold as of the *next* bar, from today's z-score."""
        if position_side == 0:
            if z_t > self.entry_z:
                return -1  # short the spread: short A, long B
            if z_t < -self.entry_z:
                return 1  # long the spread: long A, short B
            return 0
        if position_side == 1:
            exit_hit = z_t >= self.exit_z
            stop_hit = z_t <= -self.stop_loss_z
            return 0 if (exit_hit or stop_hit) else 1
        # position_side == -1
        exit_hit = z_t <= self.exit_z
        stop_hit = z_t >= self.stop_loss_z
        return 0 if (exit_hit or stop_hit) else -1

    def run(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate the strategy and return (daily_portfolio, trade_log).

        `price_a`/`price_b` are the raw prices used for execution;
        `hedge_ratio`, `spread`, and `zscore` are assumed to already carry
        whatever lag `src/engine` applied to avoid lookahead bias. Rows
        where `hedge_ratio` or `zscore` is still NaN (warm-up) are held
        flat rather than traded.
        """
        pair_name = f"{price_a.name or 'A'}/{price_b.name or 'B'}"
        aligned = pd.concat(
            [price_a, price_b, hedge_ratio, spread, zscore],
            axis=1,
            keys=["price_a", "price_b", "hedge_ratio", "spread", "zscore"],
            join="inner",
        ).sort_index()

        cash = self.initial_capital
        position_side = 0
        open_position: _OpenPosition | None = None
        pending_target = 0
        previous_equity = self.initial_capital
        realized_pnl_cum = 0.0

        portfolio_rows: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []

        for date, row in aligned.iterrows():
            price_a_t = float(row["price_a"])
            price_b_t = float(row["price_b"])
            hedge_t = row["hedge_ratio"]
            spread_t = row["spread"]
            z_t = row["zscore"]

            # --- execute the transition decided from the previous bar's z-score ---
            if pending_target != position_side:
                if open_position is not None:
                    exit_fee_a = self.transaction_fee_pct * abs(open_position.qty_a) * price_a_t
                    exit_fee_b = self.transaction_fee_pct * abs(open_position.qty_b) * price_b_t
                    exit_fees = exit_fee_a + exit_fee_b

                    gross_pnl = open_position.qty_a * (
                        price_a_t - open_position.entry_price_a
                    ) + open_position.qty_b * (price_b_t - open_position.entry_price_b)
                    total_fees = open_position.entry_fees + exit_fees
                    net_pnl = gross_pnl - total_fees

                    cash += open_position.qty_a * price_a_t
                    cash += open_position.qty_b * price_b_t
                    cash -= exit_fees

                    trades.append(
                        {
                            "entry_date": open_position.entry_date,
                            "exit_date": date,
                            "pair": pair_name,
                            "type": "LONG" if open_position.side == 1 else "SHORT",
                            "entry_price_a": open_position.entry_price_a,
                            "entry_price_b": open_position.entry_price_b,
                            "exit_price_a": price_a_t,
                            "exit_price_b": price_b_t,
                            "gross_pnl": gross_pnl,
                            "fees": total_fees,
                            "net_pnl": net_pnl,
                        }
                    )
                    realized_pnl_cum += net_pnl
                    open_position = None
                    position_side = 0

                if pending_target != 0:
                    capital_alloc = cash
                    qty_a = pending_target * (capital_alloc / price_a_t)
                    qty_b = -hedge_t * qty_a

                    entry_fee_a = self.transaction_fee_pct * abs(qty_a) * price_a_t
                    entry_fee_b = self.transaction_fee_pct * abs(qty_b) * price_b_t
                    entry_fees = entry_fee_a + entry_fee_b

                    cash -= qty_a * price_a_t
                    cash -= qty_b * price_b_t
                    cash -= entry_fees

                    open_position = _OpenPosition(
                        entry_date=date,
                        side=pending_target,
                        qty_a=qty_a,
                        qty_b=qty_b,
                        entry_price_a=price_a_t,
                        entry_price_b=price_b_t,
                        entry_fees=entry_fees,
                    )
                    position_side = pending_target

            # --- mark to market ---
            qty_a_held = open_position.qty_a if open_position else 0.0
            qty_b_held = open_position.qty_b if open_position else 0.0
            equity = cash + qty_a_held * price_a_t + qty_b_held * price_b_t
            daily_return = (equity / previous_equity - 1.0) if previous_equity else 0.0
            unrealized_pnl = (
                qty_a_held * (price_a_t - open_position.entry_price_a)
                + qty_b_held * (price_b_t - open_position.entry_price_b)
                if open_position
                else 0.0
            )

            portfolio_rows.append(
                {
                    "date": date,
                    "equity": equity,
                    "cash": cash,
                    "position": position_side,
                    "spread": spread_t,
                    "zscore": z_t,
                    "returns": daily_return,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl_cum,
                }
            )
            previous_equity = equity

            # --- decide the target position for the *next* bar ---
            if pd.isna(z_t) or pd.isna(hedge_t):
                pending_target = position_side
            else:
                pending_target = self._next_target(position_side, float(z_t))

        portfolio_df = pd.DataFrame(portfolio_rows).set_index("date")
        trade_log_df = pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)
        return portfolio_df, trade_log_df
