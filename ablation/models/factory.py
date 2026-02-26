"""
Model factory.

build_model(config) reads two fields from the config:
  - config.backbone   → selects the feature extractor
  - config.use_tabular → selects Single vs Dual branch

Everything else (feat_dim, tabular_dim, embed_dim) is inferred automatically
so callers never have to keep sizes in sync manually.
"""

from __future__ import annotations
import torch.nn as nn

from configs.config import BaseConfig, DualBranchConfig, FiLMConfig, CrossAttnConfig
from .backbones.base import build_backbone
from .single_branch import SingleBranchModel
from .dual_branch import DualBranchModel, FiLMDualBranch, CrossAttnDualBranch


def build_model(config: BaseConfig, device: str = "cpu") -> nn.Module:
    """
    Construct a model from a config object and move it to *device*.

    Returns either a SingleBranchModel or DualBranchModel depending on
    whether config.use_tabular is True.
    """
    backbone = build_backbone(
        name=config.backbone,
        pretrained=config.pretrained,
        freeze=config.freeze_backbone,
    )

    if isinstance(config, CrossAttnConfig):
        model = CrossAttnDualBranch(
            backbone,
            tabular_dim=config.tabular_dim,
            tabular_embed_dim=config.tabular_embed_dim,
            d_attn=config.d_attn,
            n_heads=config.n_heads,
        )
    elif isinstance(config, FiLMConfig):
        model = FiLMDualBranch(
            backbone,
            tabular_dim=config.tabular_dim,
            tabular_embed_dim=config.tabular_embed_dim,
        )
    elif isinstance(config, DualBranchConfig):
        model = DualBranchModel(
            backbone,
            tabular_dim=config.tabular_dim,
            tabular_embed_dim=config.tabular_embed_dim,
        )
    else:
        model = SingleBranchModel(backbone)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"  Model : {type(model).__name__}")
    print(f"  Backbone  : {config.backbone}  (feat_dim={backbone.feat_dim})")
    print(f"  Tabular   : {isinstance(config, DualBranchConfig)}")
    print(f"  Params    : {total:,} total  /  {trainable:,} trainable")
    print(f"  Pretrained: {config.pretrained}  |  frozen: {config.freeze_backbone}")
    print(f"{'='*60}\n")

    return model.to(device)
