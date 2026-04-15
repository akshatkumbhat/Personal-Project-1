from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils.helpers import load_config


class Preprocessor:
    """Cleans, normalizes, and windows time-series data."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.pre_cfg = self.config["preprocessing"]
        self.scaler = None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and basic cleaning."""
        df = df.copy()

        # Drop fully empty rows
        df.dropna(how="all", inplace=True)

        # Fill missing values
        if self.pre_cfg["fill_method"] == "ffill":
            df.ffill(inplace=True)
            df.bfill(inplace=True)  # catch leading NaNs
        elif self.pre_cfg["fill_method"] == "interpolate":
            df.interpolate(method="time", inplace=True)
            df.bfill(inplace=True)

        return df

    def scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numeric columns."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if self.pre_cfg["scaling_method"] == "zscore":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def inverse_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse scaling transformation."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call scale() first.")
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.inverse_transform(df[numeric_cols])
        return df

    def create_windows(
        self,
        data: np.ndarray,
        window_size: int | None = None,
    ) -> np.ndarray:
        """Create sliding windows for sequence models.

        Args:
            data: 2D array of shape (n_samples, n_features)
            window_size: Length of each window

        Returns:
            3D array of shape (n_windows, window_size, n_features)
        """
        window_size = window_size or self.pre_cfg["window_size"]

        if len(data) <= window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be > window_size ({window_size})"
            )

        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data[i : i + window_size])

        return np.array(windows)

    def process(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Full preprocessing pipeline: clean -> scale."""
        df = self.clean(df)
        df = self.scale(df, fit=fit)
        return df
