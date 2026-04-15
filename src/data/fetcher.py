from __future__ import annotations

import os
import pandas as pd
import yfinance as yf
from src.utils.helpers import load_config, ensure_dir


class DataFetcher:
    """Fetches OHLCV data from Yahoo Finance with local caching."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.data_cfg = self.config["data"]
        self.raw_dir = ensure_dir(self.data_cfg["raw_dir"])

    def fetch(
        self,
        ticker: str | None = None,
        period: str | None = None,
        interval: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker. Uses cache unless force=True."""
        ticker = ticker or self.data_cfg["default_ticker"]
        period = period or self.data_cfg["period"]
        interval = interval or self.data_cfg["interval"]

        cache_path = os.path.join(self.raw_dir, f"{ticker}_{interval}_{period}.csv")

        if not force and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"Loaded cached data for {ticker} ({len(df)} rows)")
            return df

        print(f"Downloading {ticker} data (period={period}, interval={interval})...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_csv(cache_path)
        print(f"Saved {len(df)} rows to {cache_path}")
        return df

    def fetch_multiple(
        self,
        tickers: list[str] | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        tickers = tickers or self.data_cfg["tickers"]
        return {t: self.fetch(ticker=t, **kwargs) for t in tickers}
