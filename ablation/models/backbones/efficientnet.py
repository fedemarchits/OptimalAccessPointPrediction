"""EfficientNet-B3 backbones — 12-channel and RGB-only variants, feat_dim = 1536."""

import torch
import torch.nn as nn

from .base import MultiChannelBackbone, ChannelAttention, _init_12ch_conv

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


class EfficientNetB3Backbone(MultiChannelBackbone):
    """
    EfficientNet-B3 modified for 12-channel satellite input.

    A ChannelAttention module is applied first so the network can learn
    which of the 12 input channels are most informative before feature
    extraction begins.  The regression head is not built; the model
    returns the 1536-d globally-pooled feature vector.
    """

    feat_dim: int = 1536

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        if not _TIMM_OK:
            raise ImportError(
                "timm is required for EfficientNet-B3.  "
                "Install with: pip install timm"
            )
        super().__init__()

        net = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",   # returns (B, 1536, H, W) feature maps
        )
        orig_conv = net.conv_stem

        # Replace first conv: 3 → 12 input channels
        net.conv_stem = nn.Conv2d(
            12, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        _init_12ch_conv(net.conv_stem, orig_conv.weight, pretrained)

        self.channel_attention = ChannelAttention(12)
        self.backbone = net

        # Global average pool + flatten: (B, 1536, H, W) → (B, 1536)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("EfficientNetB3Backbone: backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize RGB channels with ImageNet stats; DEM+LandUse left as-is
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        x = self.channel_attention(x)
        return self.pool(self.backbone(x))   # (B, 1536)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature map (B, 1536, H, W) before global pooling."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        x = self.channel_attention(x)
        return self.backbone(x)              # (B, 1536, 7, 7) for 224×224 input


class EfficientNetB3RGBBackbone(MultiChannelBackbone):
    """
    EfficientNet-B3 trained on RGB channels only (channels 0-2).

    Uses the standard pretrained ImageNet first conv unchanged — no weight
    redistribution across extra channels.  Channels 3-11 (DEM + LandUse)
    are silently discarded so the network sees only the raw satellite image.

    This is an ablation baseline: how much does removing DEM and land-use
    context hurt performance?
    """

    feat_dim: int = 1536

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        if not _TIMM_OK:
            raise ImportError(
                "timm is required for EfficientNet-B3.  "
                "Install with: pip install timm"
            )
        super().__init__()

        # Standard 3-channel model — pretrained weights fully intact
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("EfficientNetB3RGBBackbone: backbone frozen")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization so pretrained weights see expected input range."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x[:, :3] - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.backbone(self._normalize(x)))

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._normalize(x))
