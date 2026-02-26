"""
Trainer — handles the full training lifecycle for both model types.

The Trainer is model-agnostic: it uses isinstance(model, DualBranchModel)
to decide whether to forward tabular features, which is explicit and
readable without requiring models to share an interface beyond nn.Module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dual_branch import DualBranchModel
try:
    from models.dual_branch import FiLMDualBranch, CrossAttnDualBranch
except ImportError:
    FiLMDualBranch = None
    CrossAttnDualBranch = None
from training.metrics import calculate_metrics

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience  = patience
        self.counter   = 0
        self.best_loss: Optional[float] = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages training, validation, checkpointing, and test evaluation
    for a single experiment defined by *config*.

    Usage:
        trainer = Trainer(model, config, train_loader, val_loader,
                          test_loader, train_dataset)
        trainer.fit()
        trainer.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        train_dataset,               # for denormalize_target + tabular stats
    ):
        self.model        = model
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.denorm       = train_dataset.denormalize_target

        # Device
        self.device = (
            config.device
            if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Detect model type once
        _dual_types = (
            (DualBranchModel,)
            + ((FiLMDualBranch,)      if FiLMDualBranch      else ())
            + ((CrossAttnDualBranch,) if CrossAttnDualBranch else ())
        )
        self._uses_tabular = isinstance(model, _dual_types)

        # Output directory
        run_name       = getattr(config, "wandb_name", None) or "run"
        self.output_dir = Path(config.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tc = config
        # Loss, optimiser, scheduler
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )
        self.early_stopping = EarlyStopping(patience=tc.patience)

        # TensorBoard
        self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))

        # Weights & Biases
        self.use_wandb = tc.use_wandb and _WANDB_OK
        if self.use_wandb:
            wandb.init(
                project=tc.wandb_project,
                name=run_name,
                config={
                    "backbone":       tc.backbone,
                    "uses_tabular":   self._uses_tabular,
                    "pretrained":     tc.pretrained,
                    "freeze_backbone": tc.freeze_backbone,
                    "learning_rate":  tc.learning_rate,
                    "weight_decay":   tc.weight_decay,
                    "batch_size":     tc.batch_size,
                    "num_epochs":     tc.num_epochs,
                    "patience":       tc.patience,
                    "device":         self.device,
                },
            )
            wandb.watch(model, log=None)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        images  = batch["image"].to(self.device)
        targets = batch["target"].to(self.device).squeeze()
        if self._uses_tabular:
            tabular = batch["tabular"].to(self.device)
            preds   = self.model(images, tabular).squeeze()
        else:
            preds   = self.model(images).squeeze()
        return preds, targets

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        desc: str = "",
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run one full pass over *loader*.

        Returns (metrics_dict, cluster_metrics_dict).
        Gradient updates are performed only when train=True.
        """
        self.model.train() if train else self.model.eval()

        running_loss = 0.0
        all_preds:    list = []
        all_targets:  list = []
        all_clusters: list = []

        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        with grad_ctx:
            for batch in tqdm(loader, desc=desc, leave=False):
                preds, targets = self._forward(batch)
                loss = self.criterion(preds, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                running_loss += loss.item()
                all_preds.extend(np.atleast_1d(preds.detach().cpu().numpy()))
                all_targets.extend(np.atleast_1d(targets.detach().cpu().numpy()))
                all_clusters.extend(batch["metadata"]["cluster"].numpy())

        metrics = calculate_metrics(
            np.array(all_preds, dtype=np.float32),
            np.array(all_targets, dtype=np.float32),
            denormalize_fn=self.denorm,
        )
        metrics["loss"] = running_loss / len(loader)

        # Per-cluster MAE (diagnostic)
        cluster_metrics: Dict[str, float] = {}
        for cid in range(5):
            mask = np.array(all_clusters) == cid
            if mask.sum() == 0:
                continue
            cp = np.array(all_preds)[mask]
            ct = np.array(all_targets)[mask]
            cp = self.denorm(torch.from_numpy(cp)).numpy()
            ct = self.denorm(torch.from_numpy(ct)).numpy()
            cluster_metrics[f"cluster_{cid}_mae"] = float(np.mean(np.abs(cp - ct)))

        return metrics, cluster_metrics

    def _save(self, epoch: int, metrics: Dict, tag: str) -> None:
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics":             metrics,
            },
            self.output_dir / f"{tag}.pth",
        )

    def _load(self, tag: str) -> None:
        ckpt = torch.load(
            self.output_dir / f"{tag}.pth",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self) -> Dict:
        """
        Full training loop with early stopping, checkpointing,
        TensorBoard and optional W&B logging.

        Returns training history dict.
        """
        tc = self.config
        best_val_loss = float("inf")
        best_epoch    = 0
        history = {k: [] for k in
                   ["train_loss", "train_mae", "train_r2",
                    "val_loss",   "val_mae",   "val_r2"]}

        for epoch in range(tc.num_epochs):
            t0 = time.time()
            ep = epoch + 1

            train_m, _         = self._run_epoch(self.train_loader, train=True,
                                                  desc=f"Ep {ep}/{tc.num_epochs} [train]")
            val_m,   cluster_m = self._run_epoch(self.val_loader,   train=False,
                                                  desc=f"Ep {ep}/{tc.num_epochs} [val]  ")

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_m["loss"])
            new_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            print(
                f"\nEpoch {ep}/{tc.num_epochs}  "
                f"({elapsed:.1f}s  lr={new_lr:.2e})\n"
                f"  train  loss={train_m['loss']:.4f}  "
                f"MAE={train_m['mae']:.1f}  R²={train_m['r2']:.4f}\n"
                f"  val    loss={val_m['loss']:.4f}  "
                f"MAE={val_m['mae']:.1f}  R²={val_m['r2']:.4f}"
            )

            # TensorBoard
            for key in ("loss", "mae", "r2"):
                self.writer.add_scalars(
                    key.upper(),
                    {"train": train_m[key], "val": val_m[key]},
                    epoch,
                )
            for k, v in cluster_m.items():
                self.writer.add_scalar(f"val/cluster/{k}", v, epoch)

            # W&B
            if self.use_wandb:
                wandb.log({
                    "epoch":       epoch,
                    "lr":          new_lr,
                    "train/loss":  train_m["loss"],
                    "train/mae":   train_m["mae"],
                    "train/rmse":  train_m["rmse"],
                    "train/r2":    train_m["r2"],
                    "val/loss":    val_m["loss"],
                    "val/mae":     val_m["mae"],
                    "val/rmse":    val_m["rmse"],
                    "val/r2":      val_m["r2"],
                    **{f"val/{k}": v for k, v in cluster_m.items()},
                })

            # History
            for key in ("loss", "mae", "r2"):
                history[f"train_{key}"].append(float(train_m[key]))
                history[f"val_{key}"].append(float(val_m[key]))

            # Best checkpoint
            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                best_epoch    = ep
                self._save(epoch, val_m, "best_model")
                print(f"  ✓ best model saved  (val_loss={best_val_loss:.4f})")
                if self.use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"]    = best_epoch

            # Periodic checkpoint
            if ep % tc.save_every == 0:
                self._save(epoch, val_m, f"checkpoint_epoch_{ep}")

            # Early stopping
            if self.early_stopping(val_m["loss"]):
                print(f"Early stopping at epoch {ep} (best epoch: {best_epoch})")
                break

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        self.writer.close()
        return history

    def evaluate(self) -> Dict[str, float]:
        """
        Load the best checkpoint and run evaluation on the test set.
        Saves results to test_results.json and logs to W&B if enabled.
        """
        self._load("best_model")
        test_m, test_cluster_m = self._run_epoch(
            self.test_loader, train=False, desc="Test"
        )
        results = {**test_m, **test_cluster_m}

        print("\n── Test Results " + "─" * 44)
        for k, v in results.items():
            print(f"  {k:<25} {v:.4f}")
        print("─" * 60)

        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in results.items()})
            wandb.finish()

        return results
