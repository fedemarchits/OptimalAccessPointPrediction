"""
DINOv2 ViT-B/14 backbone — 12-channel feature extractor, feat_dim = 768.

DINOv2 (Oquab et al., 2023) is a ViT pretrained with self-supervised
distillation on 142M curated images.  It produces features that generalise
far better across visual domains than ImageNet-supervised CNNs, which is
exactly what we need for city-level geographic generalisation.

Key implementation choices
──────────────────────────
• Loaded via timm (`vit_base_patch14_dinov2.lvd142m`).
• Patch embedding expanded from 3 → 12 input channels using the same
  strategy as the CNN backbones: pretrained RGB weights preserved in
  channels 0-2; channels 3-11 (DEM + LandUse) initialised with
  Kaiming-normal scaled by 0.01 so they start as a tiny perturbation.
• ImageNet RGB normalisation applied inside forward() so the pretrained
  patch embedding sees values in the expected range.
• forward_spatial() reshapes the 256 patch tokens (16×16 grid) into a
  (B, 768, 16, 16) spatial map, compatible with FiLM and CrossAttn fusion.
• feat_dim = 768, matching ConvNeXt-Tiny for fair comparison.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import MultiChannelBackbone

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


class DINOv2Backbone(MultiChannelBackbone):
    """
    DINOv2 ViT-B/14 modified for 12-channel satellite input.

    Returns the globally-pooled CLS token: (B, 768).
    The forward_spatial() method returns the patch-token grid: (B, 768, 16, 16).
    """

    feat_dim: int = 768          # ViT-B/14 embed_dim
    _patch_size: int = 14        # DINOv2 patch size
    _grid_size: int = 16         # 224 / 14 = 16 patches per side

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        if not _TIMM_OK:
            raise ImportError(
                "timm is required for DINOv2.  "
                "Install with: pip install timm>=0.9.0"
            )
        super().__init__()

        # ── Load pretrained ViT-B/14 ─────────────────────────────────────────
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            in_chans=3,          # load RGB pretrained weights first
            num_classes=0,       # remove classification head → returns CLS feat
            global_pool="token", # use CLS token as global representation
            img_size=224,        # override native 518px; pos embeddings interpolated
        )

        # ── Expand patch embedding: 3 → 12 input channels ───────────────────
        orig_proj = self.backbone.patch_embed.proj   # Conv2d(3, 768, 14, 14)

        new_proj = nn.Conv2d(
            12, orig_proj.out_channels,
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride,
            padding=orig_proj.padding,
            bias=orig_proj.bias is not None,
        )

        with torch.no_grad():
            # Copy pretrained RGB weights (channels 0-2)
            if pretrained:
                new_proj.weight[:, :3] = orig_proj.weight.data
            else:
                nn.init.kaiming_normal_(new_proj.weight[:, :3])
            # Extra channels (DEM + LandUse): tiny Kaiming init
            nn.init.kaiming_normal_(new_proj.weight[:, 3:])
            new_proj.weight[:, 3:] *= 0.01
            if orig_proj.bias is not None:
                new_proj.bias.data.copy_(orig_proj.bias.data)

        self.backbone.patch_embed.proj = new_proj

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("DINOv2Backbone: backbone frozen")

    # ── Normalisation helper ──────────────────────────────────────────────────

    def _normalize_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalisation to RGB channels only (in-place safe)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x.clone()
        x[:, :3] = (x[:, :3] - mean) / std
        return x

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, H, W)
        Returns:
            (B, 768)  — CLS token after DINOv2 transformer
        """
        return self.backbone(self._normalize_rgb(x))   # (B, 768)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return patch tokens as a spatial feature map (B, 768, 16, 16).

        Used by FiLMDualBranch and CrossAttnDualBranch for spatial fusion.
        The 256 patch tokens (16×16 grid from a 224×224 / 14px patch) are
        reshaped into a 2-D spatial tensor matching the CNN backbone API.
        """
        x = self._normalize_rgb(x)

        # forward_features returns all tokens: (B, 1 + num_patches, embed_dim)
        # The first token(s) are prefix tokens (CLS, and register tokens if any).
        all_tokens = self.backbone.forward_features(x)        # (B, 1+256, 768)

        n_prefix = self.backbone.num_prefix_tokens            # typically 1 (CLS)
        patch_tokens = all_tokens[:, n_prefix:]               # (B, 256, 768)

        B, N, D = patch_tokens.shape
        H = W = int(math.isqrt(N))                           # 16
        return patch_tokens.transpose(1, 2).reshape(B, D, H, W)  # (B, 768, 16, 16)


class DINOv2RGBBackbone(MultiChannelBackbone):
    """
    DINOv2 ViT-B/14 on RGB channels only (channels 0-2 of the 12-channel input).

    Uses the standard 3-channel pretrained DINOv2 with no patch-embedding
    modification — channels 3-11 (DEM + LandUse) are silently discarded.

    Ablation baseline: how much do the extra modalities contribute vs. plain
    RGB with a stronger self-supervised backbone?
    """

    feat_dim: int = 768
    _patch_size: int = 14
    _grid_size: int = 16

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        if not _TIMM_OK:
            raise ImportError(
                "timm is required for DINOv2.  "
                "Install with: pip install timm>=0.9.0"
            )
        super().__init__()

        # Standard 3-channel model — pretrained weights fully intact
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool="token",
            img_size=224,        # override native 518px; pos embeddings interpolated
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("DINOv2RGBBackbone: backbone frozen")

    def _normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        """ImageNet normalisation for a 3-channel RGB tensor."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
        return (rgb - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, H, W) — only channels 0-2 are used
        Returns:
            (B, 768)
        """
        return self.backbone(self._normalize(x[:, :3]))

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens as a spatial feature map (B, 768, 16, 16)."""
        rgb = self._normalize(x[:, :3])
        all_tokens = self.backbone.forward_features(rgb)
        n_prefix = self.backbone.num_prefix_tokens
        patch_tokens = all_tokens[:, n_prefix:]
        B, N, D = patch_tokens.shape
        H = W = int(math.isqrt(N))
        return patch_tokens.transpose(1, 2).reshape(B, D, H, W)
