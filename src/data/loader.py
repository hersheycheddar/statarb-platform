"""Historical market data loading with local Parquet caching.

Statistical rationale: pairs-trading and cointegration analysis require a
common, gap-free price history across every leg of a pair. Silently
forward-filling long gaps or leaving misaligned calendars between tickers
would corrupt hedge-ratio and cointegration estimates downstream, so
alignment and missing-data handling are done once, centrally, here rather
than in the engine layer.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path("data/cache")
RAW_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class MarketDataLoader:
    """Fetches and caches aligned adjusted-close prices for a set of tickers.

    Raw OHLCV data for each ticker is cached to disk as Parquet (see
    SPEC.md section 1.1) so repeated backtests over overlapping date ranges
    do not re-hit the network. `fetch()` returns the wide-format Price
    Panel described in SPEC.md section 1.2.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str | date | datetime,
        end: str | date | datetime,
        interval: str = "1d",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        self.tickers: list[str] = list(dict.fromkeys(tickers))
        self.start: pd.Timestamp = pd.Timestamp(start).tz_localize("UTC") \
            if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
        self.end: pd.Timestamp = pd.Timestamp(end).tz_localize("UTC") \
            if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
        if self.start >= self.end:
            raise ValueError("start must be before end")
        self.interval = interval
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_{self.interval}.parquet"

    def _fetch_from_api(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Download raw OHLCV data for a single ticker from yfinance.

        Isolated as its own method so tests can monkeypatch it and avoid
        making real network calls.
        """
        raw = yf.download(
            ticker,
            start=start.tz_localize(None),
            end=end.tz_localize(None),
            interval=self.interval,
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            return self._empty_raw_frame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "date"
        return raw[RAW_COLUMNS]

    def _empty_raw_frame(self) -> pd.DataFrame:
        data = {col: pd.Series(dtype="float64") for col in RAW_COLUMNS}
        data["volume"] = pd.Series(dtype="int64")
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([], name="date", tz="UTC")
        return df

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        """Return raw OHLCV for `ticker` over [self.start, self.end].

        Reads from the local Parquet cache when it already fully covers the
        requested range; otherwise downloads via the API, merges the result
        into the cache, and writes it back to disk.
        """
        cache_path = self._cache_path(ticker)
        cached = self._empty_raw_frame()
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            cached.index.name = "date"

        covers_range = (
            not cached.empty
            and cached.index.min() <= self.start
            and cached.index.max() >= self.end
        )
        if covers_range:
            window = cached.loc[(cached.index >= self.start) & (cached.index <= self.end)]
            return window

        fresh = self._fetch_from_api(ticker, self.start, self.end)
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(cache_path)

        return combined.loc[(combined.index >= self.start) & (combined.index <= self.end)]

    def fetch(self) -> pd.DataFrame:
        """Return an aligned, gap-handled Price Panel of adjusted-close prices.

        Statistical rationale: cointegration and hedge-ratio estimation
        assume synchronized observations across both legs of a pair. Small
        gaps (e.g. a single-exchange holiday affecting one ticker) are
        forward-filled for up to 2 sessions; any date where a ticker still
        has no observation after that is dropped entirely so no leg is
        silently misaligned with a stale price.
        """
        series = {ticker: self._load_ticker(ticker)["adj_close"] for ticker in self.tickers}

        panel = pd.DataFrame(series)
        panel = panel.sort_index()
        panel = panel.ffill(limit=2)
        panel = panel.dropna(how="any")
        panel.columns.name = "ticker"
        panel.index.name = "date"
        return panel
