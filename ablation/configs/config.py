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

    warmup_epochs: backbone is frozen for this many epochs so the regression
    head can first adapt to ImageNet features before the full network is
    fine-tuned end-to-end with discriminative learning rates (backbone lr/10).
    """
    backbone: str = "efficientnet_b3"   # "resnet50" | "efficientnet_b3" | "convnext_tiny"
    use_tabular: bool = False
    warmup_epochs: int = 2              # freeze backbone for first N epochs


# ---------------------------------------------------------------------------
# Tabular-only (no image backbone)
# ---------------------------------------------------------------------------

@dataclass
class TabularOnlyConfig(BaseConfig):
    """
    Pure tabular MLP — no image backbone at all.
    Ablation baseline: how much can OSM features alone predict?
    """
    use_tabular: bool = True
    tabular_dim: int = 15
    warmup_epochs: int = 0              # no backbone to warm up


# ---------------------------------------------------------------------------
# RGB-only single-branch (ablation: no DEM, no land-use channels)
# ---------------------------------------------------------------------------

@dataclass
class RGBOnlyConfig(SingleBranchConfig):
    """
    Image-only model using only the 3 RGB channels.

    Ablation baseline to quantify how much the extra modalities
    (DEM + 8-channel land-use one-hot) contribute vs. plain RGB.
    Uses EfficientNet-B3 with its original pretrained ImageNet weights
    (no channel-weight redistribution).
    """
    backbone: str = "efficientnet_b3_rgb"


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
    Backbone + FiLM conditioning.
    Tabular features generate per-channel scale+shift applied to the
    backbone spatial map before global pooling.
    Requires backbone to implement forward_spatial().
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15
    tabular_embed_dim: int = 64


# ---------------------------------------------------------------------------
# Cross-attention dual-branch
# ---------------------------------------------------------------------------

@dataclass
class CrossAttnConfig(BaseConfig):
    """
    Backbone + cross-attention fusion.
    Tabular embedding queries the image spatial feature map (H×W locations)
    to identify which image regions matter most for population estimation.
    Requires backbone to implement forward_spatial().
    Compatible with efficientnet_b3 (feat_dim=1536) and convnext_tiny (feat_dim=768).
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15
    tabular_embed_dim: int = 64
    d_attn: int = 256              # cross-attention projection dimension
    n_heads: int = 8               # number of attention heads


# ---------------------------------------------------------------------------
# Multi-task dual-branch (collaborative cluster-prediction auxiliary task)
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig(DualBranchConfig):
    """
    DualBranchModel + collaborative cluster-prediction auxiliary head.

    Adds a small classifier that predicts which urban-fabric cluster (0-4)
    each sample belongs to, using the same backbone features as the main
    regression task.  The auxiliary cross-entropy loss encourages the
    backbone to learn richer urban-morphology representations.

    Unlike DANN (which adversarially removes country information), this is a
    positive multi-task setup: cluster awareness improves, not hurts, the
    main regression objective.

    cluster_loss_weight: weight α for the auxiliary loss (L = L_task + α * L_cluster).
    """
    backbone: str = "convnext_tiny"
    n_clusters: int = 5
    cluster_loss_weight: float = 0.5


# ---------------------------------------------------------------------------
# DANN dual-branch (domain-adversarial, EfficientNet-B3)
# ---------------------------------------------------------------------------

@dataclass
class DANNConfig(BaseConfig):
    """
    DualBranchModel + Domain-Adversarial Training (DANN).

    Adds a domain classifier (country prediction) with a Gradient Reversal
    Layer to the standard dual-branch architecture. The backbone is forced
    to produce country-invariant features, reducing the val→test gap caused
    by the model learning city-specific shortcuts.

    dann_max_lambda: maximum GRL scaling factor (annealed 0 → max_lambda).
    n_domains:       number of country classes (12 for this dataset).
    """
    backbone: str = "efficientnet_b3"
    use_tabular: bool = True
    tabular_dim: int = 15
    tabular_embed_dim: int = 64
    n_domains: int = 12           # number of country classes
    dann_max_lambda: float = 1.0  # GRL ramps from 0 to this over training
    warmup_epochs: int = 0        # freeze backbone for first N epochs (0 = disabled)


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
