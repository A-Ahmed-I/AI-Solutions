import timm
import torch
import torch.nn as nn
from src.constant.constant import *


class ParkinsonClassifier(nn.Module):
    """
    Hybrid classifier that fuses CNN image features with handcrafted HOG+LBP
    features for Parkinson's disease detection.

    Architecture
    ------------
    * **Image branch**  : EfficientNet-B0 (pretrained, no head)
                          → ``(B, num_features)``
    * **Math branch**   : 2-layer MLP with BatchNorm + Dropout
                          ``MATH_FEATURE_DIM → 512 → 128``
    * **Fusion head**   : 3-layer MLP
                          ``(num_features + 128) → 512 → 256 → 1``

    The output is a raw logit (apply ``torch.sigmoid`` for probability).
    """

    def __init__(self, backbone_name: str = BACKBONE_NAME) -> None:
        """
        Parameters
        ----------
        backbone_name : str
            Any ``timm``-compatible model name.  Default: ``"efficientnet_b0"``.
        """
        super().__init__()

        # Image branch
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        cnn_out_dim = self.backbone.num_features

        # Handcrafted-feature branch
        self.math_branch = nn.Sequential(
            nn.Linear(MATH_FEATURE_DIM, MATH_HIDDEN_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(MATH_HIDDEN_DIM),
            nn.Dropout(0.4),
            nn.Linear(MATH_HIDDEN_DIM, MATH_OUTPUT_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(MATH_OUTPUT_DIM),
            nn.Dropout(0.4),
        )

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_out_dim + MATH_OUTPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        math_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image         : torch.Tensor  ``(B, 3, H, W)``
        math_features : torch.Tensor  ``(B, MATH_FEATURE_DIM)``

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, 1)``.
        """
        img_feat = self.backbone(image)
        math_feat = self.math_branch(math_features)
        combined = torch.cat([img_feat, math_feat], dim=1)
        return self.fusion_head(combined)
