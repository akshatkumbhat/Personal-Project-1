import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.statistical import StatisticalDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.ensemble import EnsembleDetector


@pytest.fixture
def normal_series():
    np.random.seed(42)
    return pd.Series(np.random.randn(200), name="returns")


@pytest.fixture
def series_with_anomalies():
    np.random.seed(42)
    data = np.random.randn(200)
    # Inject obvious anomalies
    data[50] = 10.0
    data[100] = -8.0
    data[150] = 12.0
    return pd.Series(data, name="returns")


@pytest.fixture
def stat_detector():
    config = {
        "models": {
            "statistical": {
                "zscore_threshold": 3.0,
                "ewma_span": 20,
                "ewma_threshold": 3.0,
            }
        }
    }
    return StatisticalDetector(config)


class TestStatisticalDetector:
    def test_zscore_detects_outliers(self, stat_detector, series_with_anomalies):
        result = stat_detector.zscore_detect(series_with_anomalies)
        assert result["anomaly"].iloc[50]
        assert result["anomaly"].iloc[100]
        assert result["anomaly"].iloc[150]

    def test_zscore_normal_data(self, stat_detector, normal_series):
        result = stat_detector.zscore_detect(normal_series)
        # Should flag very few points in normal data
        anomaly_rate = result["anomaly"].mean()
        assert anomaly_rate < 0.05

    def test_ewma_detects_outliers(self, stat_detector):
        # EWMA needs sustained normal data before a spike to detect it
        np.random.seed(42)
        data = np.random.randn(200) * 0.5  # low variance normal data
        data[150] = 15.0  # extreme spike after baseline is established
        series = pd.Series(data, name="returns")
        result = stat_detector.ewma_detect(series)
        assert result["anomaly"].sum() > 0

    def test_predict_returns_correct_shape(self, stat_detector, normal_series):
        labels = stat_detector.predict(normal_series)
        assert len(labels) == len(normal_series)
        assert set(np.unique(labels)).issubset({-1, 1})


class TestIsolationForest:
    def test_fit_predict(self):
        config = {
            "models": {
                "isolation_forest": {
                    "n_estimators": 100,
                    "contamination": 0.05,
                    "max_samples": "auto",
                    "random_state": 42,
                }
            }
        }
        detector = IsolationForestDetector(config)

        # Normal data with injected anomalies
        np.random.seed(42)
        X = np.random.randn(200, 3)
        X[50] = [10, 10, 10]  # obvious outlier
        X[100] = [-8, -8, -8]

        detector.fit(X)
        labels = detector.predict(X)

        assert labels[50] == -1  # should flag the outlier
        assert labels[100] == -1

    def test_not_fitted_raises(self):
        config = {
            "models": {
                "isolation_forest": {
                    "n_estimators": 100,
                    "contamination": 0.05,
                    "max_samples": "auto",
                    "random_state": 42,
                }
            }
        }
        detector = IsolationForestDetector(config)
        with pytest.raises(RuntimeError):
            detector.predict(np.random.randn(10, 3))

    def test_detect_auto_fits(self):
        config = {
            "models": {
                "isolation_forest": {
                    "n_estimators": 100,
                    "contamination": 0.05,
                    "max_samples": "auto",
                    "random_state": 42,
                }
            }
        }
        detector = IsolationForestDetector(config)
        df = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        result = detector.detect(df)
        assert "anomaly_score" in result.columns
        assert "anomaly" in result.columns


class TestEnsemble:
    def test_combine_scores(self):
        config = {
            "models": {
                "ensemble": {
                    "weights": {
                        "statistical": 0.3,
                        "isolation_forest": 0.7,
                    },
                    "threshold": 0.5,
                }
            }
        }
        ens = EnsembleDetector(config)
        scores = {
            "statistical": np.array([0.2, 0.8, 0.1]),
            "isolation_forest": np.array([0.3, 0.9, 0.05]),
        }
        combined = ens.combine_scores(scores)
        assert combined.shape == (3,)
        # Weighted average check
        expected_0 = (0.3 * 0.2 + 0.7 * 0.3) / 1.0
        assert abs(combined[0] - expected_0) < 1e-6

    def test_detect_returns_dataframe(self):
        config = {
            "models": {
                "ensemble": {
                    "weights": {"a": 0.5, "b": 0.5},
                    "threshold": 0.5,
                }
            }
        }
        ens = EnsembleDetector(config)
        scores = {
            "a": np.array([0.1, 0.9, 0.3]),
            "b": np.array([0.2, 0.8, 0.4]),
        }
        result = ens.detect(scores)
        assert "ensemble_score" in result.columns
        assert "anomaly" in result.columns
        assert "n_votes" in result.columns
