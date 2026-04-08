from typing import Tuple
import torch
import torch.nn as nn


class NeuroVoxCNN(nn.Module):
    """
    Custom CNN architecture for spectrogram-based audio classification.

    Architecture:
        [Conv → BN → ReLU → MaxPool] × 3
                ↓
        Adaptive Global Average Pooling
                ↓
        Fully Connected Classification Head

    Designed for medical audio tasks (e.g., PD vs HC classification).
    """

    def __init__(
        self, input_ch: int, hidden_ch: int, out_ch: int, dropout_rate: float = 0.5
    ) -> None:
        """
        Args:
            input_ch (int):
                Number of input channels (e.g., 1 for spectrograms).
            hidden_ch (int):
                Base number of convolution filters.
            out_ch (int):
                Number of output classes.
            dropout_rate (float, optional):
                Dropout probability. Default is 0.5.
        """
        super().__init__()

        self.block1: nn.Sequential = self._block(input_ch, hidden_ch, 3)
        self.block2: nn.Sequential = self._block(hidden_ch, hidden_ch * 2, 2)
        self.block3: nn.Sequential = self._block(hidden_ch * 2, hidden_ch * 4, 3)

        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_ch * 4, hidden_ch * 2),
            nn.BatchNorm1d(hidden_ch * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ch * 2, hidden_ch),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Logits of shape (B, out_ch)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)

        logits: torch.Tensor = self.classifier(x)

        return logits

    def _block(self, input_ch: int, hidden_ch: int, kernel_size: int) -> nn.Sequential:
        """
        Convolutional block:
            Conv2D → BatchNorm → ReLU → MaxPool

        Args:
            input_ch (int):
                Number of input channels.
            hidden_ch (int):
                Number of output channels.
            kernel_size (int):
                Convolution and pooling kernel size.

        Returns:
            nn.Sequential:
                Convolutional feature extraction block.
        """
        return nn.Sequential(
            nn.Conv2d(input_ch, hidden_ch, kernel_size=kernel_size),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size),
        )
