"""
Experiment configurations for the ablation study.

Two config hierarchies:
  SingleBranchConfig  – image-only backbone models (3 backbones × 1 = 3 runs)
  DualBranchConfig    – image + OSM tabular models (3 backbones × 1 = 3 runs)

Both inherit from BaseConfig which holds every shared hyperparameter so that
diff-ing two runs is trivial and reproducibility is guaranteed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """All hyperparameters shared across every experiment."""

    # ── Data paths (must be set per experiment / environment) ───────────────
    json_file: str = ""
    base_dir: str = ""

    # ── Data preprocessing ──────────────────────────────────────────────────
    crop_size: int = 224
    jitter_pixels: int = 40          # spatial augmentation jitter (±px)
    target_clamp_max: float = 80_000.0
    normalize_targets: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    random_seed: int = 42

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    patience: int = 10              # early stopping
    save_every: int = 5             # checkpoint every N epochs
    num_workers: int = 4

    # ── Backbone ────────────────────────────────────────────────────────────
    pretrained: bool = True
    freeze_backbone: bool = False

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: str = "outputs"     # checkpoints and logs go here

    # ── Device ──────────────────────────────────────────────────────────────
    device: Optional[str] = None        # None → auto-detect (cuda if available)

    # ── W&B ─────────────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "population-prediction-ablation"
    wandb_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Single-branch (image only)
# ---------------------------------------------------------------------------

@dataclass
class SingleBranchConfig(BaseConfig):
    """
    Image-only model: backbone feature extractor + regression head.
    Corresponds to the baseline architecture (Branch 1 of the ablation).
    """
    backbone: str = "efficientnet_b3"   # "resnet50" | "efficientnet_b3" | "convnext_tiny"
    use_tabular: bool = False


# ---------------------------------------------------------------------------
# Dual-branch (image + tabular)
# ---------------------------------------------------------------------------

@dataclass
class DualBranchConfig(BaseConfig):
    """
    Image + OSM-tabular model: backbone features concatenated with a
    tabular MLP embedding, then passed to a joint fusion head.
    Corresponds to Branch 2 of the ablation.
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15            # must match N_TABULAR_FEATURES in dataset.py
    tabular_embed_dim: int = 64      # output dim of the tabular MLP branch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FiLM dual-branch (EfficientNet only)
# ---------------------------------------------------------------------------

@dataclass
class FiLMConfig(BaseConfig):
    """
    EfficientNet-B3 + FiLM conditioning.
    Tabular features generate per-channel scale+shift applied to the
    backbone spatial map before global pooling.
    Only compatible with backbone='efficientnet_b3'.
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15
    tabular_embed_dim: int = 64


# ---------------------------------------------------------------------------
# Cross-attention dual-branch (EfficientNet only)
# ---------------------------------------------------------------------------

@dataclass
class CrossAttnConfig(BaseConfig):
    """
    EfficientNet-B3 + cross-attention fusion.
    Tabular embedding queries the image spatial feature map (7×7 locations)
    to identify which image regions matter most for population estimation.
    Only compatible with backbone='efficientnet_b3'.
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15
    tabular_embed_dim: int = 64
    d_attn: int = 256              # cross-attention projection dimension
    n_heads: int = 8               # number of attention heads


# ---------------------------------------------------------------------------
# Pre-built ablation grids
# ---------------------------------------------------------------------------

SINGLE_BRANCH_EXPERIMENTS: list[SingleBranchConfig] = [
    SingleBranchConfig(backbone="resnet50",       wandb_name="single_resnet50"),
    SingleBranchConfig(backbone="efficientnet_b3", wandb_name="single_efficientnet_b3"),
    SingleBranchConfig(backbone="convnext_tiny",  wandb_name="single_convnext_tiny"),
]

DUAL_BRANCH_EXPERIMENTS: list[DualBranchConfig] = [
    DualBranchConfig(backbone="resnet50",       wandb_name="dual_resnet50"),
    DualBranchConfig(backbone="efficientnet_b3", wandb_name="dual_efficientnet_b3"),
    DualBranchConfig(backbone="convnext_tiny",  wandb_name="dual_convnext_tiny"),
]
