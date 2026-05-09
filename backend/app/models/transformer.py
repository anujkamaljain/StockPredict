"""
Temporal Fusion Transformer for stock prediction.
State-of-the-art architecture for time-series forecasting with interpretability.
Optimized for V100-32GB GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from Temporal Fusion Transformer paper.
    Provides adaptive depth and non-linear processing with skip connections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()

        self.output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(hidden_size, self.output_size * 2)  # For GLU
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

        if input_size != self.output_size:
            self.skip = nn.Linear(input_size, self.output_size)
        else:
            self.skip = None

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)

        gate_input = self.fc2(hidden)
        gate_input = self.dropout(gate_input)

        # Gated Linear Unit
        a, b = gate_input.chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)

        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network from TFT.
    Learns which input features are most important at each timestep.
    """

    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_features = num_features

        # Per-feature GRNs
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(input_size // num_features, hidden_size, hidden_size, dropout)
             for _ in range(num_features)]
        )

        # Softmax weights for feature selection
        self.weight_grn = GatedResidualNetwork(
            input_size, hidden_size, num_features, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            selected: (batch, seq_len, hidden_size)
            weights: (batch, seq_len, num_features)
        """
        feat_size = x.size(-1) // self.num_features

        # Split into individual features
        features = x.split(feat_size, dim=-1)

        # Process each feature
        processed = []
        for i, feat in enumerate(features):
            if i < len(self.feature_grns):
                processed.append(self.feature_grns[i](feat))

        if not processed:
            return x, torch.ones(x.size(0), x.size(1), 1, device=x.device)

        processed = torch.stack(processed, dim=-1)  # (batch, seq, hidden, num_feat)

        # Compute selection weights
        weights = torch.softmax(self.weight_grn(x), dim=-1)  # (batch, seq, num_feat)

        # Weighted combination
        selected = (processed * weights.unsqueeze(-2)).sum(dim=-1)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for stock prediction.

    Architecture:
        Input -> Variable Selection -> LSTM Encoder -> Multi-Head Attention
        -> GRN -> Output Heads

    Designed to fit on V100-32GB with seq_length=60 and batch_size=64.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_length: int = 60,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_length = seq_length

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # LSTM encoder for local temporal patterns
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # Gated skip connection over LSTM
        self.lstm_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # Transformer encoder for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Gated residual for transformer output
        self.transformer_grn = GatedResidualNetwork(d_model, dim_feedforward, d_model, dropout)

        # Output heads
        self.output_shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Direction probability
        self.direction_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Expected return
        self.return_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Uncertainty estimation (aleatoric)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            dict with direction_prob, return_pred, uncertainty
        """
        batch_size = x.size(0)

        # Embed input
        embedded = self.input_embedding(x)  # (batch, seq, d_model)
        embedded = self.pos_encoding(embedded)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(embedded)

        # Gated skip connection: GLU(cat(lstm, input))
        gate_input = torch.cat([lstm_out, embedded], dim=-1)
        gate = self.lstm_gate(gate_input)
        a, b = gate.chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)
        lstm_enriched = self.lstm_norm(gated + embedded)

        # Causal mask for transformer (prevents attending to future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )

        # Transformer encoding
        transformer_out = self.transformer_encoder(lstm_enriched, mask=causal_mask)

        # GRN post-processing
        enriched = self.transformer_grn(transformer_out)

        # Use last timestep for prediction
        last_step = enriched[:, -1, :]  # (batch, d_model)

        # Shared representation
        shared = self.output_shared(last_step)

        return {
            "direction_prob": self.direction_head(shared).squeeze(-1),
            "return_pred": self.return_head(shared).squeeze(-1),
            "uncertainty": self.uncertainty_head(shared).squeeze(-1),
        }
