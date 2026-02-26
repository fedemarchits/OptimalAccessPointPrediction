"""
Run test-set evaluation on an already-trained checkpoint.

Usage:
    python scripts/evaluate.py --run dual_efficientnet_b3 --model dual --backbone efficientnet_b3
    python scripts/evaluate.py --run single_resnet50      --model single --backbone resnet50
    python scripts/evaluate.py --run crossattn_efficientnet_b3 --model crossattn
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from data import get_cached_dataloaders
from models import build_model
from training import Trainer

JSON_FILE = "/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR  = "/workspace/PopulationDataset"
CACHE_DIR = "/workspace/PopulationDataset/cache"

parser = argparse.ArgumentParser()
parser.add_argument("--run",      required=True,  help="Run name (must match the outputs/ subfolder)")
parser.add_argument("--model",    required=True,  choices=["single", "dual", "crossattn"])
parser.add_argument("--backbone", default="efficientnet_b3",
                    choices=["resnet50", "efficientnet_b3", "convnext_tiny"])
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model == "single":
    from configs import SingleBranchConfig
    cfg = SingleBranchConfig(json_file=JSON_FILE, base_dir=BASE_DIR,
                              backbone=args.backbone, wandb_name=args.run)
elif args.model == "dual":
    from configs import DualBranchConfig
    cfg = DualBranchConfig(json_file=JSON_FILE, base_dir=BASE_DIR,
                            backbone=args.backbone, wandb_name=args.run)
else:
    from configs import CrossAttnConfig
    cfg = CrossAttnConfig(json_file=JSON_FILE, base_dir=BASE_DIR,
                           wandb_name=args.run)

cfg.use_wandb = False   # no W&B logging during eval-only run

train_loader, val_loader, test_loader, train_ds = get_cached_dataloaders(
    cache_dir=CACHE_DIR,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    use_tabular=cfg.use_tabular,
    crop_size=cfg.crop_size,
)

model   = build_model(cfg, device=device)
trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, train_ds)
trainer.evaluate()
