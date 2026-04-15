from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class StatisticalDetector:
    """Anomaly detection using Z-score and EWMA control charts.

    References:
        - Shewhart control chart theory
        - EWMA: Roberts (1959) "Control Chart Tests Based on Geometric Moving Averages"
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_cfg = self.config["models"]["statistical"]

    def zscore_detect(self, series: pd.Series) -> pd.DataFrame:
        """Flag points where |z-score| > threshold.

        Args:
            series: Time-series of values (e.g., returns, price)

        Returns:
            DataFrame with columns: value, zscore, anomaly, anomaly_score
        """
        threshold = self.model_cfg["zscore_threshold"]
        mean = series.mean()
        std = series.std()

        if std == 0:
            zscores = pd.Series(0.0, index=series.index)
        else:
            zscores = (series - mean) / std

        result = pd.DataFrame(
            {
                "value": series,
                "zscore": zscores,
                "anomaly_score": zscores.abs() / threshold,  # normalized 0-1+
                "anomaly": zscores.abs() > threshold,
            },
            index=series.index,
        )
        return result

    def ewma_detect(self, series: pd.Series) -> pd.DataFrame:
        """EWMA-based control chart detection.

        Uses exponentially weighted moving average and std to detect
        points that deviate beyond control limits.

        Args:
            series: Time-series of values

        Returns:
            DataFrame with columns: value, ewma, upper, lower, anomaly, anomaly_score
        """
        span = self.model_cfg["ewma_span"]
        threshold = self.model_cfg["ewma_threshold"]

        # Use shifted EWMA/std so control limits are based on data
        # *before* the current point (prevents the spike from inflating its own bounds)
        ewma = series.ewm(span=span, adjust=False).mean().shift(1)
        ewm_std = series.ewm(span=span, adjust=False).std().shift(1)

        # Fill first row (no prior data)
        ewma.iloc[0] = series.iloc[0]
        ewm_std.iloc[0] = 0.0

        upper = ewma + threshold * ewm_std
        lower = ewma - threshold * ewm_std

        deviation = (series - ewma).abs()
        safe_std = ewm_std.replace(0, np.finfo(float).eps)
        anomaly_score = deviation / (threshold * safe_std)

        result = pd.DataFrame(
            {
                "value": series,
                "ewma": ewma,
                "upper": upper,
                "lower": lower,
                "anomaly_score": anomaly_score,
                "anomaly": (series > upper) | (series < lower),
            },
            index=series.index,
        )
        return result

    def detect(self, series: pd.Series, method: str = "both") -> pd.DataFrame:
        """Run detection and return combined results.

        Args:
            series: Time-series data
            method: 'zscore', 'ewma', or 'both'

        Returns:
            DataFrame with anomaly scores and labels
        """
        if method == "zscore":
            return self.zscore_detect(series)
        elif method == "ewma":
            return self.ewma_detect(series)

        # Both: average the anomaly scores
        z_result = self.zscore_detect(series)
        e_result = self.ewma_detect(series)

        combined = pd.DataFrame(
            {
                "value": series,
                "zscore_score": z_result["anomaly_score"],
                "ewma_score": e_result["anomaly_score"],
                "anomaly_score": (
                    z_result["anomaly_score"] + e_result["anomaly_score"]
                )
                / 2,
                "anomaly": z_result["anomaly"] | e_result["anomaly"],
            },
            index=series.index,
        )
        return combined

    def fit(self, data: pd.DataFrame) -> "StatisticalDetector":
        """No-op for API compatibility. Statistical methods are non-parametric."""
        return self

    def predict(self, series: pd.Series) -> np.ndarray:
        """Return anomaly labels (-1 = anomaly, 1 = normal) for API compatibility."""
        result = self.detect(series)
        return np.where(result["anomaly"], -1, 1)
