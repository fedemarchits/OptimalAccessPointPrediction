"""
DANN training script — ConvNeXt-Tiny + domain-adversarial training.

Same setup as dann_efficientnet_b3 but using the ConvNeXt-Tiny backbone,
which was the best-performing backbone in the dual-branch ablation.

Usage on Vast.ai:
    python scripts/train_dann_convnext.py
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
    backbone          = "convnext_tiny",
    wandb_name        = "dann_convnext_tiny",
    n_domains         = 12,      # 12 European countries in the dataset
    dann_max_lambda   = 0.3,     # gentler domain pressure
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
