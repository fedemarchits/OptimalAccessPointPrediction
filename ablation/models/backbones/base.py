"""
Base class and shared utilities for all 12-channel backbone extractors.

Every backbone:
  - accepts (B, 12, H, W) input (RGB + DEM + LandUse one-hot)
  - returns (B, feat_dim) flat feature vectors
  - carries no regression head
  - exposes `feat_dim` as a class-level integer
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class MultiChannelBackbone(nn.Module, ABC):
    """Abstract base for all 12-channel feature extractors."""

    feat_dim: int  # must be set by every concrete subclass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, H, W)  — 12-channel satellite image tensor
        Returns:
            (B, feat_dim)     — flat feature vector
        """


class ChannelAttention(nn.Module):
    """
    CBAM-style channel attention: learns per-channel importance weights
    by combining average- and max-pooled channel statistics.
    Used by EfficientNet and ConvNeXt backbones.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


def _init_12ch_conv(conv: nn.Conv2d, rgb_weight: torch.Tensor, pretrained: bool) -> None:
    """
    Initialise a 12-channel first conv layer.

    Strategy:
      - channels 0-2 (RGB): copy pretrained ImageNet weights
      - channels 3-11 (DEM + LandUse): Kaiming init scaled to 10 %
        so they start as a small perturbation rather than random noise
    """
    with torch.no_grad():
        if pretrained:
            conv.weight[:, :3, :, :] = rgb_weight
            nn.init.kaiming_normal_(
                conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu"
            )
            conv.weight[:, 3:, :, :] *= 0.1
        else:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")


def build_backbone(
    name: str,
    pretrained: bool = True,
    freeze: bool = False,
) -> MultiChannelBackbone:
    """
    Factory that maps a backbone name string to a concrete backbone instance.

    Args:
        name:      "resnet50" | "efficientnet_b3" | "convnext_tiny"
        pretrained: load ImageNet weights
        freeze:     freeze all backbone parameters (only heads are trainable)
    """
    from .resnet50 import ResNet50Backbone
    from .efficientnet import EfficientNetB3Backbone, EfficientNetB3RGBBackbone
    from .convnext import ConvNeXtTinyBackbone
    from .dinov2 import DINOv2Backbone, DINOv2RGBBackbone

    registry = {
        "resnet50":            ResNet50Backbone,
        "efficientnet_b3":     EfficientNetB3Backbone,
        "efficientnet_b3_rgb": EfficientNetB3RGBBackbone,
        "convnext_tiny":       ConvNeXtTinyBackbone,
        "dinov2_vitb14":       DINOv2Backbone,
        "dinov2_vitb14_rgb":   DINOv2RGBBackbone,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(registry.keys())}"
        )
    return registry[name](pretrained=pretrained, freeze=freeze)
