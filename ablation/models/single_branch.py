"""
Single-branch (image-only) model.

Architecture:
  (B, 12, H, W)  →  MultiChannelBackbone  →  (B, feat_dim)
                                           →  RegressionHead  →  (B, 1)
"""

import torch
import torch.nn as nn
from .backbones.base import MultiChannelBackbone


class RegressionHead(nn.Module):
    """Simple MLP that maps a flat feature vector to a scalar prediction."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SingleBranchModel(nn.Module):
    """
    Backbone feature extractor + regression head.
    Tabular features are not used; this is the image-only baseline.
    """

    def __init__(self, backbone: MultiChannelBackbone):
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(backbone.feat_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 12, H, W)
        Returns:
            (B, 1) scalar predictions
        """
        return self.head(self.backbone(image))
