from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class NaiveBaselineDetector:
    """Simple threshold-based baseline detector.

    Flags any day where:
      - |daily return| exceeds a threshold, OR
      - volume z-score exceeds a threshold

    This is the "dumb" detector that any ML model should beat.
    If your ML model can't outperform this, it's not adding value.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_cfg = self.config["models"]["naive_baseline"]
        self.return_threshold = self.model_cfg["return_threshold"]
        self.volume_threshold = self.model_cfg["volume_threshold"]

    def fit(self, data: pd.DataFrame) -> NaiveBaselineDetector:
        """No-op. Baseline has no learned parameters."""
        return self

    def detect(
        self,
        df: pd.DataFrame,
        returns_col: str = "returns",
        volume_zscore_col: str = "volume_zscore",
    ) -> pd.DataFrame:
        """Detect anomalies using simple thresholds.

        Args:
            df: DataFrame with returns and volume z-score columns
            returns_col: Column name for daily returns
            volume_zscore_col: Column name for volume z-score

        Returns:
            DataFrame with anomaly_score and anomaly columns
        """
        result = pd.DataFrame(index=df.index)

        # Return-based score: how many thresholds away
        if returns_col in df.columns:
            return_score = df[returns_col].abs() / self.return_threshold
        else:
            return_score = pd.Series(0.0, index=df.index)

        # Volume-based score
        if volume_zscore_col in df.columns:
            volume_score = df[volume_zscore_col].abs() / self.volume_threshold
        else:
            volume_score = pd.Series(0.0, index=df.index)

        result["return_score"] = return_score
        result["volume_score"] = volume_score
        result["anomaly_score"] = np.maximum(return_score, volume_score)
        result["anomaly"] = result["anomaly_score"] > 1.0  # exceeds threshold

        return result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly labels (-1 = anomaly, 1 = normal)."""
        result = self.detect(df)
        return np.where(result["anomaly"], -1, 1)
