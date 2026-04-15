from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class EnsembleDetector:
    """Weighted ensemble combining multiple anomaly detectors.

    Combines anomaly scores from statistical, Isolation Forest, and
    LSTM Autoencoder models using configurable weights and a
    consensus threshold.

    Reference:
        Aggarwal, C.C. (2017) "Outlier Analysis" — Springer.
        Chapter on ensemble methods for outlier detection.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.ensemble_cfg = self.config["models"]["ensemble"]
        self.weights = self.ensemble_cfg["weights"]
        self.threshold = self.ensemble_cfg["threshold"]

    def combine_scores(
        self,
        scores: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute weighted average of anomaly scores.

        Args:
            scores: Dict mapping model name -> anomaly scores array.
                    Keys should match weight keys: 'statistical',
                    'isolation_forest', 'autoencoder'

        Returns:
            Weighted anomaly scores
        """
        total_weight = 0.0
        combined = np.zeros_like(list(scores.values())[0], dtype=float)

        for model_name, model_scores in scores.items():
            weight = self.weights.get(model_name, 0.0)
            combined += weight * np.clip(model_scores, 0, None)
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        return combined

    def detect(
        self,
        scores: dict[str, np.ndarray],
        index=None,
    ) -> pd.DataFrame:
        """Run ensemble detection.

        Args:
            scores: Dict mapping model name -> anomaly scores
            index: Optional datetime index

        Returns:
            DataFrame with per-model scores, ensemble_score, anomaly
        """
        combined = self.combine_scores(scores)

        result = pd.DataFrame(index=index)

        for model_name, model_scores in scores.items():
            result[f"{model_name}_score"] = model_scores

        result["ensemble_score"] = combined
        result["anomaly"] = combined > self.threshold

        # Count how many individual models flag each point
        vote_threshold = 0.5  # individual model threshold
        votes = np.zeros(len(combined))
        for model_scores in scores.values():
            votes += (np.array(model_scores) > vote_threshold).astype(int)
        result["n_votes"] = votes.astype(int)
        result["consensus"] = votes >= 2  # at least 2 models agree

        return result
