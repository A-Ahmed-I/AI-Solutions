import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class NeuroVoxTL(nn.Module):
    """
    NeuroVox Transfer Learning model based on ResNet-18 backbone.

    Modifications:
        - Converts first convolution layer to accept single-channel input (e.g., spectrograms).
        - Removes original fully connected layer.
        - Adds custom classifier head.

    Architecture:
        Input (1, H, W)
            ↓
        ResNet18 Backbone (feature extractor)
            ↓
        Custom Classifier:
            Linear → BatchNorm → SiLU → Dropout → Linear
    """

    def __init__(self, num_classes: int) -> None:
        """
        Args:
            num_classes (int):
                Number of output classes.
        """
        super().__init__()

        # Pretrained ResNet18 backbone
        self.backbone: nn.Module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first convolution layer for single-channel input
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3),
            bias=False,
        )

        # Extract feature size from original FC layer
        in_features: int = self.backbone.fc.in_features

        # Remove original classifier
        self.backbone.fc = nn.Identity()

        # Custom classification head
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, 1, H, W)

        Returns:
            torch.Tensor:
                Logits of shape (B, num_classes)
        """
        embeddings: torch.Tensor = self.backbone(x)
        logits: torch.Tensor = self.classifier(embeddings)

        return logits
