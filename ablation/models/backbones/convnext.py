"""ConvNeXt-Tiny backbone — 12-channel feature extractor, feat_dim = 768."""

import torch
import torch.nn as nn

from .base import MultiChannelBackbone, ChannelAttention, _init_12ch_conv

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


class ConvNeXtTinyBackbone(MultiChannelBackbone):
    """
    ConvNeXt-Tiny modified for 12-channel satellite input.

    Same ChannelAttention front-end as EfficientNet.  After pooling,
    a LayerNorm is applied (aligned with ConvNeXt's internal design
    which uses LN throughout rather than BN).  Returns 768-d features.
    """

    feat_dim: int = 768

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        if not _TIMM_OK:
            raise ImportError(
                "timm is required for ConvNeXt-Tiny.  "
                "Install with: pip install timm"
            )
        super().__init__()

        net = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="",   # returns (B, 768, H, W) feature maps
        )
        orig_conv = net.stem[0]

        # Replace first conv: 3 → 12 input channels
        net.stem[0] = nn.Conv2d(
            12, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        _init_12ch_conv(net.stem[0], orig_conv.weight, pretrained)

        self.channel_attention = ChannelAttention(12)
        self.backbone = net

        # Pool + LayerNorm: (B, 768, H, W) → (B, 768)
        # LN is important for ConvNeXt stability (it uses LN instead of BN)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.feat_dim),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("ConvNeXtTinyBackbone: backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize RGB channels with ImageNet stats; DEM+LandUse left as-is
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        x = self.channel_attention(x)
        return self.pool(self.backbone(x))   # (B, 768)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature map (B, 768, H, W) before global pooling."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        x = self.channel_attention(x)
        return self.backbone(x)              # (B, 768, 7, 7) for 224×224 input
