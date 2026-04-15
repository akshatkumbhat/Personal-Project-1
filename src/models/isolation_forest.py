from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.utils.helpers import load_config


class IsolationForestDetector:
    """Anomaly detection using Isolation Forest.

    Reference:
        Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008)
        "Isolation Forest" — IEEE International Conference on Data Mining.

    The key insight: anomalies are few and different, so they are
    isolated in fewer splits (shorter path length) in random trees.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_cfg = self.config["models"]["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=self.model_cfg["n_estimators"],
            contamination=self.model_cfg["contamination"],
            max_samples=self.model_cfg["max_samples"],
            random_state=self.model_cfg["random_state"],
        )
        self._is_fitted = False

    def fit(self, data: pd.DataFrame | np.ndarray) -> "IsolationForestDetector":
        """Fit the Isolation Forest model.

        Args:
            data: Training data (n_samples, n_features). Should be mostly normal data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.model.fit(data)
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Returns:
            Array of -1 (anomaly) or 1 (normal)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if isinstance(data, pd.DataFrame):
            data = data.values
        return self.model.predict(data)

    def score_samples(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return anomaly scores. Lower = more anomalous.

        Returns:
            Array of anomaly scores (negative = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if isinstance(data, pd.DataFrame):
            data = data.values
        return self.model.score_samples(data)

    def detect(
        self, data: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Run full detection pipeline.

        Args:
            data: DataFrame with features
            feature_cols: Columns to use. If None, uses all numeric columns.

        Returns:
            DataFrame with anomaly_score and anomaly columns added
        """
        if feature_cols:
            X = data[feature_cols].values
        else:
            X = data.select_dtypes(include=[np.number]).values

        if not self._is_fitted:
            self.fit(X)

        scores = self.score_samples(X)
        labels = self.predict(X)

        # Normalize scores to 0-1 range (higher = more anomalous)
        normalized_scores = (scores.max() - scores) / (
            scores.max() - scores.min() + np.finfo(float).eps
        )

        result = data.copy()
        result["anomaly_score"] = normalized_scores
        result["anomaly"] = labels == -1
        return result
