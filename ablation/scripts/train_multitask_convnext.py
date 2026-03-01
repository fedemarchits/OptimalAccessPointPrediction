"""
Multi-task training script — ConvNeXt-Tiny + collaborative cluster-prediction.

Extends the best-performing dual-branch model (dual_convnext_tiny) with an
auxiliary classification head that predicts the urban-fabric cluster (0-4).

Unlike DANN (which adversarially removes country information), this is a
*positive* multi-task setup: the auxiliary loss encourages the backbone to
learn richer urban-morphology features (residential vs commercial vs industrial
etc.), which directly benefits population estimation.

Total loss: L_task + cluster_loss_weight * L_cluster
  L_task    = HuberLoss(population_pred, population_target)
  L_cluster = CrossEntropyLoss(cluster_pred, cluster_label)

Usage on Vast.ai:
    python scripts/train_multitask_convnext.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs import MultiTaskConfig
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

JSON_FILE = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR  = "/workspace/PopulationDataset"
CACHE_DIR = "/workspace/PopulationDataset/cache"

cfg = MultiTaskConfig(
    backbone             = "convnext_tiny",
    json_file            = JSON_FILE,
    base_dir             = BASE_DIR,
    wandb_name           = "multitask_convnext_tiny",
    n_clusters           = 5,
    cluster_loss_weight  = 0.5,
    num_epochs           = 100,
    batch_size           = 32,
    learning_rate        = 1e-4,
    weight_decay         = 5e-5,
    patience             = 7,
)

device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_loader, val_loader, test_loader, train_ds = get_cached_dataloaders(
    cache_dir=CACHE_DIR,
    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    use_tabular=cfg.use_tabular, crop_size=cfg.crop_size,
)

model   = build_model(cfg, device=device)
trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, train_ds)
trainer.fit()
trainer.evaluate()
