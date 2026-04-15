from __future__ import annotations

import time
import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class StreamSimulator:
    """Simulates a real-time data stream by replaying historical data.

    Can optionally inject synthetic anomalies (spikes, drift, level shifts)
    to test detector robustness.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.stream_cfg = self.config["stream"]
        self._data = None
        self._index = 0
        self._injected_labels = None

    def load(self, df: pd.DataFrame) -> "StreamSimulator":
        """Load historical data for replay."""
        self._data = df.copy()
        self._index = 0
        self._injected_labels = np.zeros(len(df), dtype=bool)
        return self

    def reset(self):
        """Reset stream to beginning."""
        self._index = 0

    @property
    def is_exhausted(self) -> bool:
        return self._data is None or self._index >= len(self._data)

    def _inject_anomaly(self, row: pd.Series) -> tuple[pd.Series, bool]:
        """Optionally inject a synthetic anomaly into a data point."""
        if not self.stream_cfg["inject_anomalies"]:
            return row, False

        if np.random.random() > self.stream_cfg["anomaly_probability"]:
            return row, False

        row = row.copy()
        anomaly_type = np.random.choice(self.stream_cfg["anomaly_types"])

        if anomaly_type == "spike" and "Close" in row.index:
            direction = np.random.choice([-1, 1])
            magnitude = row["Close"] * np.random.uniform(0.05, 0.15)
            row["Close"] += direction * magnitude

        elif anomaly_type == "drift" and "Close" in row.index:
            row["Close"] *= np.random.uniform(1.03, 1.08)

        elif anomaly_type == "level_shift" and "Volume" in row.index:
            row["Volume"] *= np.random.uniform(3.0, 8.0)

        return row, True

    def next(self) -> tuple[pd.Series, bool] | None:
        """Get next data point from stream.

        Returns:
            Tuple of (data_point, is_injected_anomaly) or None if exhausted
        """
        if self.is_exhausted:
            return None

        row = self._data.iloc[self._index]
        row, is_anomaly = self._inject_anomaly(row)
        self._injected_labels[self._index] = is_anomaly
        self._index += 1
        return row, is_anomaly

    def stream(self, realtime: bool = False):
        """Generator that yields data points one at a time.

        Args:
            realtime: If True, adds delay between points to simulate real-time
        """
        while not self.is_exhausted:
            result = self.next()
            if result is None:
                break

            yield result

            if realtime:
                time.sleep(1.0 / self.stream_cfg["replay_speed"])

    def get_injected_labels(self) -> np.ndarray:
        """Return boolean array of which points had injected anomalies."""
        return self._injected_labels[: self._index].copy()

    def get_progress(self) -> float:
        """Return stream progress as fraction 0-1."""
        if self._data is None:
            return 0.0
        return self._index / len(self._data)
