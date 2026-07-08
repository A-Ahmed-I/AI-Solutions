import torch
import timm
import torch.nn as nn


class MultimodalGatedModel(nn.Module):
    """
    Multimodal neural network for Parkinson's classification.

    Combines:
        - Image features (CNN backbone via timm)
        - Handcrafted numerical features

    Architecture:
        1. Image encoder (CNN backbone)
        2. Feature encoder (MLP)
        3. Gating mechanism (controls feature contribution)
        4. Fusion head (final classifier)

    Output:
        - Single logit (use BCEWithLogitsLoss)
    """

    def __init__(
        self, backbone_name: str = "efficientnet_b0", feature_dim: int = 300
    ) -> None:
        """
        Parameters
        ----------
        backbone_name : str
            Name of timm model (e.g., efficientnet_b0)
        feature_dim : int
            Dimension of handcrafted input features
        """
        super().__init__()

        self.image_backbone = timm.create_model(
            model_name=backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        backbone_output_dim = self.image_backbone.num_features

        self.feature_norm = nn.LayerNorm(feature_dim)

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        self.gating_network = nn.Sequential(
            nn.Linear(backbone_output_dim + 128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_dim + 128, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        image : torch.Tensor
            Shape (B, 3, H, W)
        features : torch.Tensor
            Shape (B, feature_dim)

        Returns
        -------
        torch.Tensor
            Shape (B, 1) — raw logits
        """

        image_features = self.image_backbone(image)  # (B, F_img)

        features = self.feature_norm(features)
        feature_embedding = self.feature_encoder(features)  # (B, 128)

        combined_features = torch.cat([image_features, feature_embedding], dim=1)
        gate = self.gating_network(combined_features)  # (B, 128)

        gated_features = feature_embedding * gate

        fused = torch.cat([image_features, gated_features], dim=1)
        logits = self.classifier(fused)

        return logits
