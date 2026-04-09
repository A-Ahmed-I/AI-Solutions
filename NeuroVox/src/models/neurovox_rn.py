import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic Residual Block (ResNet-style).

    Structure:
        Conv → BN → ReLU
        Conv → BN
        + Shortcut (identity or projection)
        → ReLU

    Used for stable deep feature learning with skip connections.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            stride (int, optional): Stride for spatial downsampling.
                                   Default is 1.
        """
        super().__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, stride=stride
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_ch)
        self.relu: nn.ReLU = nn.ReLU()

        self.conv2: nn.Conv2d = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_ch)

        # Shortcut connection
        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residual block.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor after residual addition.
        """
        identity: torch.Tensor = self.shortcut(x)

        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)

        return out


class NeuroVoxRN(nn.Module):
    """
    Custom Residual Network for spectrogram-based audio classification.

    Architecture:
        Initial Conv
            ↓
        Residual Block × 3 with progressive channel expansion
            ↓
        Global Average Pooling
            ↓
        Fully Connected Classifier

    Designed for medical audio classification tasks.
    """

    def __init__(
        self, input_ch: int, hidden_ch: int, out_ch: int, dropout_rate: float = 0.5
    ) -> None:
        """
        Args:
            input_ch (int): Number of input channels (e.g., 1).
            hidden_ch (int): Base channel width.
            out_ch (int): Number of output classes.
            dropout_rate (float, optional): Dropout probability.
        """
        super().__init__()

        # Initial convolution
        self.conv1: nn.Conv2d = nn.Conv2d(input_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(hidden_ch)
        self.relu: nn.ReLU = nn.ReLU()

        # Residual layers
        self.layer1: ResidualBlock = ResidualBlock(hidden_ch, hidden_ch)
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(2)

        self.layer2: ResidualBlock = ResidualBlock(hidden_ch, hidden_ch * 2)
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(2)

        self.layer3: ResidualBlock = ResidualBlock(hidden_ch * 2, hidden_ch * 4)

        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ch * 4, hidden_ch * 2),
            nn.BatchNorm1d(hidden_ch * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ch * 2, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, out_ch)
        """
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.layer3(x)

        x = self.global_pool(x)

        logits: torch.Tensor = self.classifier(x)

        return logits
