import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.feature_engineer import FeatureEngineer
from src.detection.stream_simulator import StreamSimulator
from src.evaluation.metrics import AnomalyEvaluator


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = np.random.randn(n).cumsum() + 100
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.5,
            "High": close + abs(np.random.randn(n)),
            "Low": close - abs(np.random.randn(n)),
            "Close": close,
            "Volume": np.random.randint(1e6, 1e7, n).astype(float),
        },
        index=dates,
    )


class TestFeatureEngineer:
    def test_engineer_adds_features(self, sample_ohlcv):
        config = {
            "features": {
                "returns": True,
                "log_returns": True,
                "volatility_window": 20,
                "rsi_period": 14,
                "bollinger_window": 20,
                "bollinger_std": 2,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "volume_zscore_window": 20,
                "rolling_stats_window": 20,
            }
        }
        fe = FeatureEngineer(config)
        result = fe.engineer(sample_ohlcv)

        expected_features = [
            "returns", "log_returns", "volatility", "rsi",
            "bb_pct", "macd_hist", "atr", "volume_zscore",
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

        # No NaNs after drop
        assert not result.isna().any().any()

    def test_get_feature_columns(self):
        fe = FeatureEngineer({"features": {}})
        cols = fe.get_feature_columns()
        assert len(cols) > 0
        assert "returns" in cols


class TestStreamSimulator:
    def test_stream_replays_data(self, sample_ohlcv):
        config = {
            "stream": {
                "replay_speed": 100,
                "inject_anomalies": False,
                "anomaly_types": ["spike"],
                "anomaly_probability": 0.0,
            }
        }
        sim = StreamSimulator(config)
        sim.load(sample_ohlcv)

        count = 0
        for row, is_anomaly in sim.stream():
            count += 1
            assert not is_anomaly  # injection disabled
        assert count == len(sample_ohlcv)

    def test_inject_anomalies(self, sample_ohlcv):
        config = {
            "stream": {
                "replay_speed": 100,
                "inject_anomalies": True,
                "anomaly_types": ["spike"],
                "anomaly_probability": 1.0,  # always inject
            }
        }
        sim = StreamSimulator(config)
        sim.load(sample_ohlcv)

        injected = 0
        for row, is_anomaly in sim.stream():
            if is_anomaly:
                injected += 1
        assert injected > 0

    def test_progress(self, sample_ohlcv):
        config = {
            "stream": {
                "replay_speed": 100,
                "inject_anomalies": False,
                "anomaly_types": [],
                "anomaly_probability": 0.0,
            }
        }
        sim = StreamSimulator(config)
        sim.load(sample_ohlcv)
        assert sim.get_progress() == 0.0
        sim.next()
        assert sim.get_progress() > 0.0


class TestAnomalyEvaluator:
    def test_compute_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        metrics = AnomalyEvaluator.compute_metrics(y_true, y_pred)
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert metrics["n_true_anomalies"] == 3
        assert metrics["n_predicted_anomalies"] == 2

    def test_detection_latency(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        latency = AnomalyEvaluator.detection_latency(y_true, y_pred)
        assert latency["n_total_events"] == 2
        assert latency["n_detected"] == 2

    def test_label_known_events(self):
        dates = pd.date_range("2020-02-15", periods=30, freq="D")
        df = pd.DataFrame({"Close": range(30)}, index=dates)
        events = [
            {"name": "Test", "ticker": "SPY", "start": "2020-02-20", "end": "2020-02-25"}
        ]
        labels = AnomalyEvaluator.label_known_events(df, events, "SPY")
        assert labels.sum() == 6  # Feb 20-25 inclusive
        assert labels[0] == 0  # Feb 15

    def test_label_points(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "returns": [0.01, -0.005, 0.04, -0.06, 0.002, 0.01, -0.035, 0.005, 0.0, 0.02],
            "volume_zscore": [0.5, 1.0, 0.3, 2.0, 0.1, 3.5, 0.2, 0.1, 4.0, 0.5],
        }, index=dates)
        labels = AnomalyEvaluator.label_points(df, min_abs_return=0.03, min_volume_zscore=3.0)
        # Day 2 (0.04), Day 3 (-0.06), Day 5 (vol 3.5), Day 6 (-0.035), Day 8 (vol 4.0)
        assert labels[2] == 1   # |0.04| > 0.03
        assert labels[3] == 1   # |-0.06| > 0.03
        assert labels[5] == 1   # vol z 3.5 > 3.0
        assert labels[0] == 0   # normal
        assert labels[4] == 0   # normal

    def test_temporal_split(self):
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        df = pd.DataFrame({"val": range(365)}, index=dates)
        train, test = AnomalyEvaluator.temporal_split(df, "2023-06-30", "2023-07-01")
        assert train.index.max() <= pd.Timestamp("2023-06-30")
        assert test.index.min() >= pd.Timestamp("2023-07-01")
        assert len(train) + len(test) == len(df)
