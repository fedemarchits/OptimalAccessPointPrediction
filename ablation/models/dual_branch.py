"""
Dual-branch (image + tabular) models.

Two variants:

  DualBranchModel — feature-level fusion (concat + MLP):
    image  → backbone → (B, feat_dim)       ──┐
                                               ├─ cat → FusionHead → (B, 1)
    tabular → TabularMLP → (B, embed_dim)   ──┘

  CrossAttnDualBranch — cross-attention fusion (EfficientNet only):
    image  → backbone (no pool) → (B, 1536, 7, 7) spatial map
    tabular → TabularMLP → (B, embed_dim) ── query
    CrossAttentionFusion: tabular queries image spatial locations
    output: cat(global_pool, attended, tab_emb) → FusionHead → (B, 1)
"""

import torch
import torch.nn as nn
from .backbones.base import MultiChannelBackbone


class TabularMLP(nn.Module):
    """
    Small MLP that encodes raw (z-scored log1p) OSM tabular features
    into a dense embedding vector.
    """

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FusionHead(nn.Module):
    """
    MLP that maps the concatenated image + tabular features to a scalar.
    Uses BatchNorm for stable training with the mixed-modality input.
    """

    def __init__(self, img_dim: int, tab_dim: int):
        super().__init__()
        in_dim = img_dim + tab_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img_feats: torch.Tensor, tab_feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([img_feats, tab_feats], dim=1))


class DualBranchModel(nn.Module):
    """
    Image + OSM-tabular model with feature-level fusion.

    The backbone is shared with SingleBranchModel — same weights,
    same initialisation — ensuring a fair comparison in the ablation study.
    """

    def __init__(
        self,
        backbone: MultiChannelBackbone,
        tabular_dim: int = 15,
        tabular_embed_dim: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.tabular_mlp = TabularMLP(tabular_dim, tabular_embed_dim)
        self.fusion = FusionHead(backbone.feat_dim, tabular_embed_dim)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   (B, 12, H, W)
            tabular: (B, tabular_dim)  — z-scored log1p OSM features
        Returns:
            (B, 1) scalar predictions
        """
        img_feats = self.backbone(image)            # (B, feat_dim)
        tab_feats = self.tabular_mlp(tabular)       # (B, tabular_embed_dim)
        return self.fusion(img_feats, tab_feats)    # (B, 1)


# ---------------------------------------------------------------------------
# Cross-attention fusion (EfficientNet-B3 only)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention where the tabular embedding (query) attends over the
    EfficientNet spatial feature map (keys / values).

    Intuition: the OSM neighbourhood context (road density, building
    coverage, POI counts…) guides the model to focus on the image
    regions that are most informative for population estimation — e.g.
    residential blocks vs commercial zones vs green space.

    Args:
        img_dim:  spatial feature channels (1536 for EfficientNet-B3)
        tab_dim:  tabular embedding dimension
        d_attn:   internal projection dimension (must be divisible by n_heads)
        n_heads:  number of attention heads
    """

    def __init__(
        self,
        img_dim: int,
        tab_dim: int,
        d_attn: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()
        assert d_attn % n_heads == 0, "d_attn must be divisible by n_heads"

        self.q_proj  = nn.Linear(tab_dim, d_attn)
        self.k_proj  = nn.Linear(img_dim, d_attn)
        self.v_proj  = nn.Linear(img_dim, d_attn)
        self.attn    = nn.MultiheadAttention(d_attn, n_heads,
                                             batch_first=True, dropout=0.1)
        self.norm_q  = nn.LayerNorm(d_attn)
        self.norm_kv = nn.LayerNorm(d_attn)
        self.out_norm = nn.LayerNorm(d_attn)

    def forward(
        self,
        spatial_feats: torch.Tensor,   # (B, N, img_dim)  N = H×W locations
        tab_emb: torch.Tensor,         # (B, tab_dim)
    ) -> torch.Tensor:                 # (B, d_attn)
        q = self.norm_q(self.q_proj(tab_emb).unsqueeze(1))   # (B, 1, d_attn)
        k = self.norm_kv(self.k_proj(spatial_feats))          # (B, N, d_attn)
        v = self.v_proj(spatial_feats)                         # (B, N, d_attn)
        attended, _ = self.attn(q, k, v)                      # (B, 1, d_attn)
        return self.out_norm(attended.squeeze(1))              # (B, d_attn)


# ---------------------------------------------------------------------------
# FiLM conditioning (EfficientNet-B3 only)
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """
    Generates per-channel scale (γ) and shift (β) from tabular features.

    The tabular embedding is projected to 2 × feature_dim values.
    γ and β are then applied element-wise to the spatial feature map:
        modulated[b, c, h, w] = γ[b,c] * spatial[b,c,h,w] + β[b,c]

    Using a residual initialisation (γ≈1, β≈0 at start) keeps training
    stable and lets the network start from the same point as a plain
    image-only baseline.
    """

    def __init__(self, tabular_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * 2),
        )
        # Initialise final layer so γ≈1 and β≈0 at the start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tabular: torch.Tensor):
        out   = self.net(tabular)                    # (B, 2 * feature_dim)
        gamma, beta = out.chunk(2, dim=-1)           # each (B, feature_dim)
        gamma = gamma + 1.0                          # residual: γ starts at 1
        return gamma, beta                           # (B, C), (B, C)


class FiLMDualBranch(nn.Module):
    """
    EfficientNet-B3 + FiLM (Feature-wise Linear Modulation) fusion.

    The OSM tabular features generate per-channel scale and shift
    parameters that modulate the EfficientNet spatial feature map
    *before* global average pooling.  This is complementary to
    CrossAttnDualBranch: FiLM controls *what features matter* (channel-
    wise recalibration) while cross-attention controls *where to look*
    (spatial attention).

    Architecture:
        image   → EfficientNet-B3 (no pool) → (B, 1536, 7, 7)
        tabular → FiLMGenerator             → γ (B,1536), β (B,1536)
        modulated = γ * spatial + β         → (B, 1536, 7, 7)
        global avg pool                     → (B, 1536)
        concat with tabular embedding       → (B, 1600)
        FusionHead                          → (B, 1)
    """

    uses_tabular: bool = True

    def __init__(
        self,
        backbone,
        tabular_dim: int = 15,
        tabular_embed_dim: int = 64,
    ):
        super().__init__()
        img_dim = 1536   # EfficientNet-B3

        self.backbone    = backbone
        self.tabular_mlp = TabularMLP(tabular_dim, tabular_embed_dim)
        self.film_gen    = FiLMGenerator(tabular_embed_dim, img_dim)

        fusion_in = img_dim + tabular_embed_dim   # 1536 + 64 = 1600
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   (B, 12, H, W)
            tabular: (B, tabular_dim)
        Returns:
            (B, 1)
        """
        spatial  = self.backbone.forward_spatial(image)     # (B, 1536, 7, 7)
        tab_emb  = self.tabular_mlp(tabular)                # (B, 64)
        gamma, beta = self.film_gen(tab_emb)                # (B, 1536), (B, 1536)

        # Apply FiLM: channel-wise scale + shift on the spatial map
        modulated    = gamma.unsqueeze(-1).unsqueeze(-1) * spatial \
                     + beta.unsqueeze(-1).unsqueeze(-1)     # (B, 1536, 7, 7)
        global_feats = modulated.mean(dim=[2, 3])           # (B, 1536)

        combined = torch.cat([global_feats, tab_emb], dim=1)  # (B, 1600)
        return self.fusion(combined)                           # (B, 1)


class CrossAttnDualBranch(nn.Module):
    """
    EfficientNet-B3 + cross-attention dual-branch model.

    The final representation concatenates three complementary views:
      1. Global average pooled image features   (B, 1536)  — "what is in the image overall"
      2. Cross-attended image features          (B, d_attn) — "what the OSM context focuses on"
      3. Tabular embedding                      (B, tab_embed_dim) — "raw neighbourhood stats"

    This is an ablation variant designed to be compared directly against
    DualBranchModel (same backbone, same tabular branch, richer fusion).
    """

    uses_tabular: bool = True   # read by Trainer to route the forward pass

    def __init__(
        self,
        backbone,                        # EfficientNetB3Backbone
        tabular_dim: int = 15,
        tabular_embed_dim: int = 64,
        d_attn: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()
        img_dim = 1536   # EfficientNet-B3 feature channels

        self.backbone     = backbone
        self.tabular_mlp  = TabularMLP(tabular_dim, tabular_embed_dim)
        self.cross_attn   = CrossAttentionFusion(img_dim, tabular_embed_dim,
                                                 d_attn, n_heads)

        fusion_in = img_dim + d_attn + tabular_embed_dim   # 1536 + 256 + 64 = 1856
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   (B, 12, H, W)
            tabular: (B, tabular_dim)
        Returns:
            (B, 1)
        """
        # EfficientNet spatial map (no global pool)
        spatial = self.backbone.forward_spatial(image)          # (B, 1536, H, W)
        B, C, H, W = spatial.shape
        spatial_seq  = spatial.flatten(2).transpose(1, 2)       # (B, H*W, 1536)
        global_feats = spatial.mean(dim=[2, 3])                 # (B, 1536)

        tab_feats = self.tabular_mlp(tabular)                   # (B, tab_embed_dim)
        attended  = self.cross_attn(spatial_seq, tab_feats)     # (B, d_attn)

        combined = torch.cat([global_feats, attended, tab_feats], dim=1)  # (B, 1856)
        return self.fusion(combined)                            # (B, 1)
