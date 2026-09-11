"""Core statistical arbitrage engine: cointegration testing, hedge-ratio
estimation, spread/z-score construction, and half-life estimation.

Statistical rationale: pairs trading assumes two price series share a
long-run equilibrium relationship (cointegration) so that some linear
combination of them is stationary and mean-reverting. Every rolling
statistic here that will later drive a trading decision is lagged so a
value "as of" bar t reflects only information available up to bar t-1 —
see CLAUDE.md's no-lookahead-bias rule and SPEC.md section 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class CointegrationResult:
    """Result of an Engle-Granger two-step cointegration test."""

    t_statistic: float
    p_value: float
    critical_values: dict[str, float]


class PairsAnalyzer:
    """Statistical toolkit for identifying and modeling a mean-reverting pair."""

    def test_cointegration(self, series_a: pd.Series, series_b: pd.Series) -> CointegrationResult:
        """Run the Engle-Granger two-step cointegration test.

        Statistical rationale: tests whether a linear combination of two
        (individually non-stationary) price series is itself stationary —
        the necessary condition for a pairs-trading spread to be
        mean-reverting rather than a spurious relationship between two
        unrelated random walks.
        """
        aligned_a, aligned_b = series_a.align(series_b, join="inner")
        aligned = pd.concat([aligned_a, aligned_b], axis=1).dropna()
        if len(aligned) < 3:
            raise ValueError("need at least 3 overlapping observations to test cointegration")

        t_statistic, p_value, critical_values = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return CointegrationResult(
            t_statistic=float(t_statistic),
            p_value=float(p_value),
            critical_values={
                "1%": float(critical_values[0]),
                "5%": float(critical_values[1]),
                "10%": float(critical_values[2]),
            },
        )

    def calculate_hedge_ratio(
        self, series_a: pd.Series, series_b: pd.Series, window: int
    ) -> pd.Series:
        """Estimate a rolling OLS hedge ratio: Series_A = beta * Series_B + alpha.

        Statistical rationale: the hedge ratio defines how many units of B
        offset one unit of A so their combination is stationary; a rolling
        (rather than static) estimate lets the relationship adapt as the
        pair's dynamics drift. beta is computed via the closed-form OLS
        slope Cov(A, B) / Var(B) over the trailing `window`, then shifted
        by one bar: the value attached to timestamp t is the beta that was
        estimated using data through t-1, so it never depends on bar t's
        own prices.
        """
        if window < 2:
            raise ValueError("window must be at least 2")
        aligned_a, aligned_b = series_a.align(series_b, join="inner")

        rolling_cov = aligned_a.rolling(window).cov(aligned_b)
        rolling_var = aligned_b.rolling(window).var()
        beta = (rolling_cov / rolling_var).shift(1)
        beta.name = "hedge_ratio"
        return beta

    def calculate_spread(
        self, series_a: pd.Series, series_b: pd.Series, hedge_ratios: pd.Series
    ) -> pd.Series:
        """Compute the pairs spread: Spread = Series_A - hedge_ratio * Series_B."""
        combined = pd.concat(
            [series_a, series_b, hedge_ratios], axis=1, keys=["a", "b", "beta"], join="inner"
        )
        spread = combined["a"] - combined["beta"] * combined["b"]
        spread.name = "spread"
        return spread

    def calculate_zscore(self, spread: pd.Series, window: int) -> pd.Series:
        """Standardize the spread against its own trailing rolling distribution.

        Statistical rationale: the z-score expresses how many rolling
        standard deviations the current spread sits from its trailing
        mean — the entry/exit signal for a mean-reversion strategy. Both
        the rolling mean and standard deviation are trailing windows
        ending at t, so no observation beyond t is used.
        """
        if window < 2:
            raise ValueError("window must be at least 2")
        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        zscore = (spread - rolling_mean) / rolling_std
        zscore.name = "zscore"
        return zscore

    def calculate_half_life(self, spread: pd.Series) -> float:
        """Estimate the mean-reversion half-life (in bars) of an OU process.

        Statistical rationale: models the spread as a discretized
        Ornstein-Uhlenbeck process, delta_spread_t = lambda * spread_{t-1}
        + alpha + epsilon. Regressing the change in spread on its own
        lagged level recovers lambda; a more negative lambda implies
        faster mean reversion. half_life = -ln(2) / lambda is the number
        of bars for a deviation to decay to half its size, which can be
        used to size expected holding periods or to filter out pairs that
        revert too slowly to trade profitably. A non-negative lambda means
        the series is not mean-reverting, so half-life is undefined
        (returned as +inf).
        """
        clean_spread = spread.dropna()
        if len(clean_spread) < 3:
            raise ValueError("need at least 3 observations to estimate half-life")

        lagged = clean_spread.shift(1)
        delta = clean_spread - lagged
        regression_data = pd.concat([delta, lagged], axis=1, keys=["delta", "lag"]).dropna()

        design = sm.add_constant(regression_data["lag"])
        model = sm.OLS(regression_data["delta"], design).fit()
        lam = model.params["lag"]

        if lam >= 0:
            return float("inf")

        return float(-np.log(2) / lam)
