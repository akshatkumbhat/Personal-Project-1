from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.helpers import load_config


class LSTMAutoencoder:
    """LSTM-based Autoencoder for time-series anomaly detection.

    Architecture:
        Encoder: Input -> LSTM(hidden) -> LSTM(encoding_dim)
        Decoder: RepeatVector -> LSTM(encoding_dim) -> LSTM(hidden) -> TimeDistributed(Dense)

    Anomalies are detected by high reconstruction error (MSE).

    Reference:
        Malhotra, P. et al. (2016) "LSTM-based Encoder-Decoder for
        Multi-Sensor Anomaly Detection" — ICML Workshop.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model_cfg = self.config["models"]["autoencoder"]
        self.model = None
        self.threshold = None
        self._n_features = None

    def _build_model(self, n_features: int):
        """Build the LSTM Autoencoder architecture."""
        # Lazy import to avoid TF startup cost when not needed
        from tensorflow import keras

        seq_len = self.model_cfg["sequence_length"]
        encoding_dim = self.model_cfg["encoding_dim"]
        hidden_dim = self.model_cfg["hidden_dim"]

        # Encoder
        inputs = keras.Input(shape=(seq_len, n_features))
        encoded = keras.layers.LSTM(hidden_dim, return_sequences=True)(inputs)
        encoded = keras.layers.LSTM(encoding_dim, return_sequences=False)(encoded)

        # Decoder
        decoded = keras.layers.RepeatVector(seq_len)(encoded)
        decoded = keras.layers.LSTM(encoding_dim, return_sequences=True)(decoded)
        decoded = keras.layers.LSTM(hidden_dim, return_sequences=True)(decoded)
        outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(decoded)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.model_cfg["learning_rate"]
            ),
            loss="mse",
        )
        self._n_features = n_features

    def fit(self, X: np.ndarray, verbose: int = 1) -> "LSTMAutoencoder":
        """Train the autoencoder on normal data.

        Args:
            X: 3D array (n_samples, sequence_length, n_features)
            verbose: Keras verbosity level
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X.shape}")

        n_features = X.shape[2]
        self._build_model(n_features)

        self.model.fit(
            X,
            X,  # autoencoder: input = target
            epochs=self.model_cfg["epochs"],
            batch_size=self.model_cfg["batch_size"],
            validation_split=self.model_cfg["validation_split"],
            verbose=verbose,
        )

        # Set threshold based on training reconstruction error
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, self.model_cfg["threshold_percentile"])

        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input sequences."""
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() first.")
        return self.model.predict(X, verbose=0)

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        reconstructions = self.reconstruct(X)
        return np.mean(np.power(X - reconstructions, 2), axis=(1, 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly labels (-1 = anomaly, 1 = normal)."""
        if self.threshold is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        errors = self.compute_reconstruction_error(X)
        return np.where(errors > self.threshold, -1, 1)

    def detect(self, X: np.ndarray, index=None) -> pd.DataFrame:
        """Run detection and return scored results.

        Args:
            X: 3D array of windowed sequences
            index: Optional datetime index for the results

        Returns:
            DataFrame with reconstruction_error, anomaly_score, anomaly
        """
        errors = self.compute_reconstruction_error(X)

        # Normalize to 0-1
        if self.threshold > 0:
            anomaly_scores = errors / self.threshold
        else:
            anomaly_scores = errors

        result = pd.DataFrame(
            {
                "reconstruction_error": errors,
                "anomaly_score": anomaly_scores,
                "anomaly": errors > self.threshold,
            }
        )

        if index is not None:
            result.index = index[: len(result)]

        return result

    def save(self, path: str):
        """Save model weights."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)

    def load(self, path: str, n_features: int):
        """Load model weights."""
        from tensorflow import keras

        self._build_model(n_features)
        self.model = keras.models.load_model(path)
