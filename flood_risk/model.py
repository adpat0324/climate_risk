"""PyTorch modules for flood risk forecasting."""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Multi-step LSTM forecaster for discharge prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
