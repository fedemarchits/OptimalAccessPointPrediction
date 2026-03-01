"""ResNet50 backbone — 12-channel feature extractor, feat_dim = 2048."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from .base import MultiChannelBackbone, _init_12ch_conv


class ResNet50Backbone(MultiChannelBackbone):
    """
    ResNet50 modified for 12-channel satellite input.

    The fc classification head is removed; the model returns the
    2048-d global-average-pooled feature vector directly.
    """

    feat_dim: int = 2048

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        orig_conv = net.conv1

        # Replace first conv: 3 → 12 input channels
        net.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _init_12ch_conv(net.conv1, orig_conv.weight, pretrained)

        # Feature extractor = everything up to (and including) avgpool;
        # fc head is intentionally excluded.
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
            net.avgpool,   # → (B, 2048, 1, 1)
        )

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            print("ResNet50Backbone: backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize RGB channels with ImageNet stats; DEM+LandUse left as-is
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        return self.features(x).flatten(1)   # (B, 2048)
