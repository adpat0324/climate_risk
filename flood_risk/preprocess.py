"""Data preprocessing utilities for flood risk modeling."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class TimeSeriesScaler:
    """Wrapper around MinMaxScaler for time series matrices."""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        shape = data.shape
        flattened = data.reshape(shape[0], -1)
        scaled = self.scaler.fit_transform(flattened)
        return scaled.reshape(shape)

    def transform(self, data: np.ndarray) -> np.ndarray:
        shape = data.shape
        flattened = data.reshape(shape[0], -1)
        scaled = self.scaler.transform(flattened)
        return scaled.reshape(shape)

    def inverse_transform(self, data: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        flattened = data.reshape(data.shape[0], -1)
        restored = self.scaler.inverse_transform(flattened)
        return restored.reshape(original_shape)


def train_val_test_split(
    x: np.ndarray, y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split sequences into train/validation/test segments."""

    n = len(x)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    logger.info(
        "Split %d samples into train=%d, val=%d, test=%d",
        n,
        len(x_train),
        len(x_val),
        len(x_test),
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def compute_risk_score(discharge: pd.Series, threshold: float) -> pd.Series:
    """Compute flood risk as exceedance probability over a discharge threshold."""

    risk = discharge / threshold
    return risk.clip(lower=0)
