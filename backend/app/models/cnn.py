"""
1D CNN model for pattern extraction in time-series data.
Multi-scale convolutional architecture captures patterns at different temporal resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualBlock1D(nn.Module):
    """Residual block with 1D convolution."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for temporal pattern extraction.

    Uses parallel convolutions with different kernel sizes to capture
    short-term (3-day), medium-term (7-day), and long-term (15-day) patterns.

    Architecture:
        Input -> Multi-Scale Conv Branches -> Concatenate -> Residual Blocks
        -> Global Pooling -> FC -> Output Heads
    """

    def __init__(
        self,
        input_size: int,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        channels = channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [3, 7, 15]

        # Multi-scale parallel branches
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_size, channels[0], ks, padding=ks // 2),
                nn.BatchNorm1d(channels[0]),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            self.branches.append(branch)

        concat_channels = channels[0] * len(kernel_sizes)

        # Deeper processing after concatenation
        layers = []
        in_ch = concat_channels
        for out_ch in channels[1:]:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            # Add residual block
            layers.append(ResidualBlock1D(out_ch, 3, dropout))
            in_ch = out_ch

        self.deep_layers = nn.Sequential(*layers)

        # Squeeze-and-Excitation (channel attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1] // 4),
            nn.GELU(),
            nn.Linear(channels[-1] // 4, channels[-1]),
            nn.Sigmoid(),
        )

        # Global pooling: both avg and max
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Output heads
        fc_input = channels[-1] * 2  # avg + max pooling

        self.shared = nn.Sequential(
            nn.Linear(fc_input, channels[-1]),
            nn.LayerNorm(channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Linear(channels[-1] // 2, 1),
            nn.Sigmoid(),
        )

        self.return_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Linear(channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            dict with direction_prob, return_pred
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale branches
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)

        # Deep layers
        x = self.deep_layers(x)

        # Channel attention
        se_weights = self.se(x).unsqueeze(-1)
        x = x * se_weights

        # Global pooling
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Output
        shared = self.shared(pooled)

        return {
            "direction_prob": self.direction_head(shared).squeeze(-1),
            "return_pred": self.return_head(shared).squeeze(-1),
        }
