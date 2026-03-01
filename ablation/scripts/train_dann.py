"""
DANN training script — EfficientNet-B3 + domain-adversarial training.

The model is identical to dual_efficientnet_b3 (DualBranchModel) but adds:
  - A domain classifier that predicts the country of each sample
  - A Gradient Reversal Layer (GRL) between the backbone and the domain head
  - A λ schedule that anneals the domain loss weight from 0 → 1 over training

The goal is to produce backbone features that are informative for population
prediction but uninformative about which country the sample comes from,
reducing the val→test generalisation gap.

Usage on Vast.ai:
    python scripts/train_dann.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import DANNConfig
from data.dataset import get_cached_dataloaders
from models import build_model
from training.trainer import Trainer

# ── Paths (edit for your environment) ────────────────────────────────────────
CACHE_DIR = "/workspace/PopulationDataset/cache"

# ── Config ───────────────────────────────────────────────────────────────────
cfg = DANNConfig(
    backbone          = "efficientnet_b3",
    wandb_name        = "dann_efficientnet_b3",
    n_domains         = 12,      # 12 European countries in the dataset
    dann_max_lambda   = 0.3,     # reduced from 1.0 — gentler domain pressure
    num_epochs        = 100,
    batch_size        = 32,
    learning_rate     = 1e-4,
    weight_decay      = 5e-5,
    patience          = 7,
)

# ── Run ───────────────────────────────────────────────────────────────────────
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

train_loader, val_loader, test_loader, train_ds = get_cached_dataloaders(
    CACHE_DIR,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    use_tabular=True,
)

# Verify n_domains matches the dataset
n_countries = train_ds.n_countries
if n_countries != cfg.n_domains:
    print(f"  Updating n_domains: {cfg.n_domains} → {n_countries}")
    cfg.n_domains = n_countries

model   = build_model(cfg, device)
trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, train_ds)

history = trainer.fit()
trainer.evaluate()
