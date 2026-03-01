"""Ablation arm: single-branch DINOv2 ViT-B/14 (image only, 12 channels)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs import SingleBranchConfig
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

JSON_FILE = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR  = "/workspace/PopulationDataset"
CACHE_DIR = "/workspace/PopulationDataset/cache"

cfg = SingleBranchConfig(
    json_file=JSON_FILE,
    base_dir=BASE_DIR,
    backbone="dinov2_vitb14",
    wandb_name="single_dinov2_vitb14",
    warmup_epochs=3,      # one extra warmup epoch — ViT-B is larger than CNNs
    num_epochs=100,
    patience=5,
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
