"""
Ablation arm: DANN dual-branch with DINOv2 ViT-B/14.

Combines the best-performing fusion strategy (DANN) with the strongest
backbone (DINOv2).  This is the primary model for the CV component.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs import DANNConfig
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

JSON_FILE = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR  = "/workspace/PopulationDataset"
CACHE_DIR = "/workspace/PopulationDataset/cache"

cfg = DANNConfig(
    json_file=JSON_FILE,
    base_dir=BASE_DIR,
    backbone="dinov2_vitb14",
    wandb_name="dann_dinov2_vitb14",
    warmup_epochs=3,
    dann_max_lambda=0.3,   # same as best-performing DANN run
    n_domains=12,
    num_epochs=100,
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
