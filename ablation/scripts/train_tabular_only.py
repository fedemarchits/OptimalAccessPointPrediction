"""Ablation arm: tabular-only MLP (no image backbone, OSM features only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs import TabularOnlyConfig
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

CACHE_DIR = "/workspace/PopulationDataset/cache"

cfg = TabularOnlyConfig(
    wandb_name="tabular_only",
    patience=7,
    use_tabular=True,
)

device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, train_ds = get_cached_dataloaders(
    cache_dir=CACHE_DIR,
    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    use_tabular=True, crop_size=cfg.crop_size,
)

model   = build_model(cfg, device=device)
trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, train_ds)
trainer.fit()
trainer.evaluate()
