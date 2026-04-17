from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


class AnomalyEvaluator:
    """Evaluation metrics for anomaly detection models.

    Designed for honest assessment of unsupervised anomaly detection:
    - Point-level labeling based on actual market behavior (not date ranges)
    - Temporal train/test split to prevent look-ahead bias
    - Baseline comparison to prove ML models add value
    - Caveats about the inherent subjectivity of anomaly labels
    """

    @staticmethod
    def label_points(
        df: pd.DataFrame,
        min_abs_return: float | None = None,
        min_volume_zscore: float | None = None,
        returns_col: str = "returns",
        volume_zscore_col: str = "volume_zscore",
        config: dict | None = None,
    ) -> np.ndarray:
        """Create point-level ground truth labels based on actual data.

        A day is labeled anomalous ONLY if:
          - |daily return| > min_abs_return (e.g., 3%), OR
          - |volume z-score| > min_volume_zscore (e.g., 3.0)

        Thresholds are read from config["evaluation"]["point_label"] if not
        provided explicitly.

        Args:
            df: DataFrame with returns and volume z-score columns
            min_abs_return: Minimum absolute return to flag as anomalous
            min_volume_zscore: Minimum volume z-score to flag as anomalous
            config: Optional config dict to read defaults from

        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        if config is not None:
            point_cfg = config.get("evaluation", {}).get("point_label", {})
        else:
            point_cfg = {}
        if min_abs_return is None:
            min_abs_return = point_cfg.get("min_abs_return", 0.03)
        if min_volume_zscore is None:
            min_volume_zscore = point_cfg.get("min_volume_zscore", 3.0)
        labels = np.zeros(len(df), dtype=int)

        if returns_col in df.columns:
            labels |= (df[returns_col].abs() > min_abs_return).astype(int).values

        if volume_zscore_col in df.columns:
            labels |= (df[volume_zscore_col].abs() > min_volume_zscore).astype(int).values

        return labels

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        train_end: str,
        test_start: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by time to prevent look-ahead bias.

        Args:
            df: DataFrame with datetime index
            train_end: Last date for training (inclusive)
            test_start: First date for testing (inclusive)

        Returns:
            (train_df, test_df)
        """
        train = df[df.index <= pd.Timestamp(train_end)]
        test = df[df.index >= pd.Timestamp(test_start)]
        return train, test

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute precision, recall, F1, and accuracy."""
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": (y_true == y_pred).mean(),
            "n_true_anomalies": int(y_true.sum()),
            "n_predicted_anomalies": int(y_pred.sum()),
            "anomaly_rate_true": float(y_true.mean()),
            "anomaly_rate_pred": float(y_pred.mean()),
        }

    @staticmethod
    def compute_roc(y_true: np.ndarray, scores: np.ndarray) -> dict:
        """Compute ROC curve and AUC."""
        if len(np.unique(y_true)) < 2:
            return {"fpr": [], "tpr": [], "auc": 0.5}

        # Clip NaN/Inf scores
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute confusion matrix components."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "matrix": cm,
        }

    @staticmethod
    def detection_latency(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        event_ranges: list[tuple[int, int]] | None = None,
    ) -> dict:
        """Measure how quickly anomalies are detected."""
        if event_ranges is None:
            event_ranges = []
            in_event = False
            start = 0
            for i, val in enumerate(y_true):
                if val == 1 and not in_event:
                    start = i
                    in_event = True
                elif val == 0 and in_event:
                    event_ranges.append((start, i))
                    in_event = False
            if in_event:
                event_ranges.append((start, len(y_true)))

        latencies = []
        for start, end in event_ranges:
            detected = False
            for i in range(start, min(end, len(y_pred))):
                if y_pred[i] == 1:
                    latencies.append(i - start)
                    detected = True
                    break
            if not detected:
                latencies.append(-1)

        detected_latencies = [lat for lat in latencies if lat >= 0]

        return {
            "mean_latency": np.mean(detected_latencies) if detected_latencies else -1,
            "max_latency": max(detected_latencies) if detected_latencies else -1,
            "min_latency": min(detected_latencies) if detected_latencies else -1,
            "n_detected": len(detected_latencies),
            "n_missed": len([lat for lat in latencies if lat < 0]),
            "n_total_events": len(event_ranges),
            "per_event_latency": latencies,
        }

    def compare_models(
        self,
        y_true: np.ndarray,
        model_predictions: dict[str, np.ndarray],
        model_scores: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Compare multiple models side by side."""
        results = []
        for name, y_pred in model_predictions.items():
            metrics = self.compute_metrics(y_true, y_pred)
            metrics["model"] = name

            if model_scores and name in model_scores:
                roc = self.compute_roc(y_true, model_scores[name])
                metrics["auc"] = roc["auc"]

            latency = self.detection_latency(y_true, y_pred)
            metrics["mean_latency"] = latency["mean_latency"]
            metrics["detection_rate"] = (
                latency["n_detected"] / max(latency["n_total_events"], 1)
            )

            results.append(metrics)

        return pd.DataFrame(results).set_index("model")

    @staticmethod
    def label_known_events(
        df: pd.DataFrame,
        events: list[dict],
        ticker: str,
    ) -> np.ndarray:
        """Create ground truth labels from known market events.

        NOTE: This is the old approach kept for backward compatibility.
        Prefer label_points() for honest evaluation.
        """
        labels = np.zeros(len(df), dtype=int)
        for event in events:
            if event.get("ticker", ticker) != ticker:
                continue
            start = pd.Timestamp(event["start"])
            end = pd.Timestamp(event["end"])
            mask = (df.index >= start) & (df.index <= end)
            labels[mask] = 1
        return labels
