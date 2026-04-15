import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import Preprocessor


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000, 10000, 100).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def preprocessor():
    config = {
        "preprocessing": {
            "fill_method": "ffill",
            "scaling_method": "zscore",
            "window_size": 10,
        }
    }
    return Preprocessor(config)


def test_clean_handles_nan(preprocessor, sample_df):
    sample_df.iloc[5, 0] = np.nan
    sample_df.iloc[10, 1] = np.nan
    cleaned = preprocessor.clean(sample_df)
    assert not cleaned.isna().any().any()


def test_scale_zscore(preprocessor, sample_df):
    scaled = preprocessor.scale(sample_df)
    # Z-scored data should have mean ~0 and std ~1
    assert abs(scaled["Close"].mean()) < 0.1
    assert abs(scaled["Close"].std() - 1.0) < 0.1


def test_scale_minmax(sample_df):
    config = {
        "preprocessing": {
            "fill_method": "ffill",
            "scaling_method": "minmax",
            "window_size": 10,
        }
    }
    prep = Preprocessor(config)
    scaled = prep.scale(sample_df)
    assert scaled["Close"].min() >= -0.01
    assert scaled["Close"].max() <= 1.01


def test_create_windows(preprocessor):
    data = np.random.randn(50, 3)
    windows = preprocessor.create_windows(data, window_size=10)
    assert windows.shape == (41, 10, 3)


def test_create_windows_too_short(preprocessor):
    data = np.random.randn(5, 3)
    with pytest.raises(ValueError):
        preprocessor.create_windows(data, window_size=10)


def test_inverse_scale(preprocessor, sample_df):
    original_close = sample_df["Close"].values.copy()
    scaled = preprocessor.scale(sample_df)
    restored = preprocessor.inverse_scale(scaled)
    np.testing.assert_array_almost_equal(restored["Close"].values, original_close, decimal=5)
