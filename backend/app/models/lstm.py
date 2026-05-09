"""
LSTM / GRU model for time-series stock prediction.
Bidirectional architecture with attention mechanism for improved sequence modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class AttentionLayer(nn.Module):
    """Temporal attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size), attention_weights: (batch, seq_len)
        """
        weights = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(weights, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with attention for stock direction prediction.

    Architecture:
        Input -> Bidirectional LSTM -> Attention -> FC -> Output

    Outputs:
        - direction_prob: P(price goes up) [0, 1]
        - return_pred: expected return (regression head)
        - volatility_pred: expected volatility (risk head)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.num_directions

        # Attention
        self.attention = AttentionLayer(lstm_out_size)

        # Output heads
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Direction prediction head (classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Return prediction head (regression)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Volatility prediction head (always positive)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size)
            return_attention: whether to return attention weights

        Returns:
            dict with 'direction_prob', 'return_pred', 'volatility_pred',
            and optionally 'attention_weights'
        """
        # Input projection
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)

        # Attention
        context, attn_weights = self.attention(lstm_out)

        # Shared representation
        shared = self.shared_fc(context)

        # Multi-task outputs
        result = {
            "direction_prob": self.direction_head(shared).squeeze(-1),
            "return_pred": self.return_head(shared).squeeze(-1),
            "volatility_pred": self.volatility_head(shared).squeeze(-1),
        }

        if return_attention:
            result["attention_weights"] = attn_weights

        return result


class GRUModel(nn.Module):
    """GRU variant — lighter alternative to LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = AttentionLayer(hidden_size * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.input_proj(x)
        out, _ = self.gru(x)
        context, _ = self.attention(out)
        return {"direction_prob": self.head(context).squeeze(-1)}
