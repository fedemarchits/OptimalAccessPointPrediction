"""Ablation arm: dual-branch EfficientNet-B3 (image + OSM tabular)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs import DualBranchConfig
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

JSON_FILE = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR  = "/workspace/PopulationDataset"
CACHE_DIR = "/workspace/PopulationDataset/cache"

cfg = DualBranchConfig(
    json_file=JSON_FILE,
    base_dir=BASE_DIR,
    backbone="efficientnet_b3",
    wandb_name="dual_efficientnet_b3",
    patience=7,
)

device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, train_ds = get_cached_dataloaders(
    cache_dir=CACHE_DIR,
    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    use_tabular=cfg.use_tabular, crop_size=cfg.crop_size,
)

model   = build_model(cfg, device=device)
trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, train_ds)
trainer.fit()
trainer.evaluate()
