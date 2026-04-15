from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class FeatureEngineer:
    """Computes technical indicators and statistical features for anomaly detection."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.feat_cfg = self.config["features"]

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple and log returns."""
        if self.feat_cfg["returns"]:
            df["returns"] = df["Close"].pct_change()
        if self.feat_cfg["log_returns"]:
            df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility (std of returns)."""
        window = self.feat_cfg["volatility_window"]
        if "returns" not in df.columns:
            df["returns"] = df["Close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=window).std()
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index."""
        period = self.feat_cfg["rsi_period"]
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands and %B indicator."""
        window = self.feat_cfg["bollinger_window"]
        num_std = self.feat_cfg["bollinger_std"]

        sma = df["Close"].rolling(window=window).mean()
        std = df["Close"].rolling(window=window).std()

        df["bb_upper"] = sma + num_std * std
        df["bb_lower"] = sma - num_std * std
        df["bb_mid"] = sma
        # %B: where price sits relative to bands (0=lower, 1=upper)
        band_width = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / band_width.replace(
            0, np.finfo(float).eps
        )
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        fast = self.feat_cfg["macd_fast"]
        slow = self.feat_cfg["macd_slow"]
        signal = self.feat_cfg["macd_signal"]

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range — measures volatility."""
        period = self.feat_cfg["atr_period"]
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume z-score and VWAP."""
        window = self.feat_cfg["volume_zscore_window"]

        # Volume z-score
        vol_mean = df["Volume"].rolling(window=window).mean()
        vol_std = df["Volume"].rolling(window=window).std()
        df["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std.replace(
            0, np.finfo(float).eps
        )

        # VWAP (cumulative within rolling window)
        df["vwap"] = (df["Close"] * df["Volume"]).rolling(window=window).sum() / (
            df["Volume"].rolling(window=window).sum().replace(0, np.finfo(float).eps)
        )
        return df

    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std, skewness, kurtosis of returns."""
        window = self.feat_cfg["rolling_stats_window"]
        if "returns" not in df.columns:
            df["returns"] = df["Close"].pct_change()

        df["roll_mean"] = df["returns"].rolling(window=window).mean()
        df["roll_std"] = df["returns"].rolling(window=window).std()
        df["roll_skew"] = df["returns"].rolling(window=window).skew()
        df["roll_kurt"] = df["returns"].rolling(window=window).kurt()
        return df

    def engineer(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        df = df.copy()
        df = self.add_returns(df)
        df = self.add_volatility(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_macd(df)
        df = self.add_atr(df)
        df = self.add_volume_features(df)
        df = self.add_rolling_stats(df)

        if drop_na:
            df.dropna(inplace=True)

        return df

    def get_feature_columns(self) -> list[str]:
        """Return list of engineered feature column names."""
        return [
            "returns",
            "log_returns",
            "volatility",
            "rsi",
            "bb_pct",
            "macd_hist",
            "atr",
            "volume_zscore",
            "roll_mean",
            "roll_std",
            "roll_skew",
            "roll_kurt",
        ]
