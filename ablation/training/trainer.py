"""
Trainer — handles the full training lifecycle for both model types.

The Trainer is model-agnostic: it uses isinstance(model, DualBranchModel)
to decide whether to forward tabular features, which is explicit and
readable without requiring models to share an interface beyond nn.Module.
"""

from __future__ import annotations

import json
import math
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
    from models.dual_branch import FiLMDualBranch, CrossAttnDualBranch, DANNDualBranch, MultiTaskDualBranch
except ImportError:
    FiLMDualBranch = None
    CrossAttnDualBranch = None
    DANNDualBranch = None
    MultiTaskDualBranch = None
try:
    from models.tabular_only import TabularOnlyModel
except ImportError:
    TabularOnlyModel = None
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
            + ((DANNDualBranch,)      if DANNDualBranch       else ())
        )
        self._uses_tabular_only = TabularOnlyModel is not None and isinstance(model, TabularOnlyModel)
        self._uses_tabular   = isinstance(model, _dual_types)
        self._uses_dann      = DANNDualBranch      is not None and isinstance(model, DANNDualBranch)
        self._uses_multitask = MultiTaskDualBranch is not None and isinstance(model, MultiTaskDualBranch)

        if self._uses_dann:
            self.domain_criterion = nn.CrossEntropyLoss()
        if self._uses_multitask:
            self.cluster_criterion = nn.CrossEntropyLoss()

        # Output directory
        run_name       = getattr(config, "wandb_name", None) or "run"
        self.output_dir = Path(config.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tc = config
        # Loss, optimiser, scheduler
        self.criterion = nn.HuberLoss(delta=1.0)

        # Backbone warm-up + discriminative LRs
        self._warmup_epochs = getattr(config, "warmup_epochs", 0)
        self._has_backbone  = hasattr(model, "backbone") and not self._uses_tabular_only

        if self._has_backbone and self._warmup_epochs > 0:
            # Freeze backbone for warm-up phase
            for p in model.backbone.parameters():
                p.requires_grad = False
            backbone_ids  = {id(p) for p in model.backbone.parameters()}
            head_params   = [p for p in model.parameters() if id(p) not in backbone_ids]
            backbone_params = list(model.backbone.parameters())
            self.optimizer = optim.AdamW([
                {"params": backbone_params, "lr": tc.learning_rate * 0.01},
                {"params": head_params,     "lr": tc.learning_rate},
            ], weight_decay=tc.weight_decay)
            print(f"  Warm-up: backbone frozen for {self._warmup_epochs} epoch(s), "
                  f"then backbone lr={tc.learning_rate*0.01:.2e} / head lr={tc.learning_rate:.2e}")
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
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
                    "backbone":        getattr(tc, "backbone", "none"),
                    "uses_tabular":    self._uses_tabular,
                    "pretrained":      getattr(tc, "pretrained", False),
                    "freeze_backbone": getattr(tc, "freeze_backbone", False),
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

    def _get_lambda(self, epoch: int) -> float:
        """DANN λ schedule: smooth ramp from 0 to dann_max_lambda over training."""
        max_lambda = getattr(self.config, "dann_max_lambda", 1.0)
        p = epoch / max(self.config.num_epochs - 1, 1)
        return float(max_lambda * (2 / (1 + math.exp(-10 * p)) - 1))

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = batch["target"].to(self.device).squeeze()
        if self._uses_tabular_only:
            tabular = batch["tabular"].to(self.device)
            preds   = self.model(tabular).squeeze()
        elif self._uses_tabular:
            images  = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            preds   = self.model(images, tabular).squeeze()
        else:
            images  = batch["image"].to(self.device)
            preds   = self.model(images).squeeze()
        return preds, targets

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        desc: str = "",
        epoch: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run one full pass over *loader*.

        Returns (metrics_dict, cluster_metrics_dict).
        Gradient updates are performed only when train=True.
        When the model is a DANNDualBranch, domain loss is added during
        training and domain classifier accuracy is tracked.
        """
        self.model.train() if train else self.model.eval()

        current_lambda      = self._get_lambda(epoch) if (train and self._uses_dann) else 0.0
        running_loss        = 0.0
        running_domain_loss = 0.0
        running_cluster_loss = 0.0
        domain_correct      = 0
        domain_total        = 0
        all_preds:         list = []
        all_targets:       list = []
        all_clusters:      list = []
        all_city_log_means: list = []
        all_city_log_stds:  list = []

        grad_ctx = torch.enable_grad() if train else torch.no_grad()

        with grad_ctx:
            for batch in tqdm(loader, desc=desc, leave=False):

                if self._uses_dann:
                    images  = batch["image"].to(self.device)
                    targets = batch["target"].to(self.device).squeeze()
                    tabular = batch["tabular"].to(self.device)

                    if train:
                        country = batch["country_label"].to(self.device)
                        preds, domain_logits = self.model(images, tabular, current_lambda)
                        preds     = preds.squeeze()
                        task_loss = self.criterion(preds, targets)
                        dom_loss  = self.domain_criterion(domain_logits, country)
                        loss      = task_loss + current_lambda * dom_loss

                        running_domain_loss += dom_loss.item()
                        domain_correct += (domain_logits.argmax(1) == country).sum().item()
                        domain_total   += country.size(0)
                        running_loss   += task_loss.item()
                    else:
                        preds, _ = self.model(images, tabular, 0.0)
                        preds    = preds.squeeze()
                        loss     = self.criterion(preds, targets)
                        running_loss += loss.item()

                elif self._uses_multitask:
                    images  = batch["image"].to(self.device)
                    targets = batch["target"].to(self.device).squeeze()
                    tabular = batch["tabular"].to(self.device)

                    if train:
                        clusters  = batch["metadata"]["cluster"].to(self.device)
                        preds, cluster_logits = self.model.forward_multitask(images, tabular)
                        preds     = preds.squeeze()
                        task_loss = self.criterion(preds, targets)
                        aux_loss  = self.cluster_criterion(cluster_logits, clusters)
                        alpha     = getattr(self.config, "cluster_loss_weight", 0.5)
                        loss      = task_loss + alpha * aux_loss
                        running_cluster_loss += aux_loss.item()
                        running_loss         += task_loss.item()
                    else:
                        preds = self.model(images, tabular).squeeze()
                        loss  = self.criterion(preds, targets)
                        running_loss += loss.item()

                else:
                    preds, targets = self._forward(batch)
                    loss = self.criterion(preds, targets)
                    running_loss += loss.item()

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                all_preds.extend(np.atleast_1d(preds.detach().cpu().numpy()))
                all_targets.extend(np.atleast_1d(targets.detach().cpu().numpy()))
                all_clusters.extend(batch["metadata"]["cluster"].numpy())
                all_city_log_means.extend(batch["city_log_mean"].cpu().numpy().flatten())
                all_city_log_stds.extend(batch["city_log_std"].cpu().numpy().flatten())

        # Build per-sample denorm closure: z → expm1(z * city_std + city_mean)
        _c_means = np.array(all_city_log_means, dtype=np.float32)
        _c_stds  = np.array(all_city_log_stds,  dtype=np.float32)
        def _denorm(t: torch.Tensor) -> torch.Tensor:
            return torch.expm1(t * torch.from_numpy(_c_stds) + torch.from_numpy(_c_means))

        metrics = calculate_metrics(
            np.array(all_preds, dtype=np.float32),
            np.array(all_targets, dtype=np.float32),
            denormalize_fn=_denorm,
        )
        metrics["loss"] = running_loss / len(loader)

        if self._uses_dann and train:
            metrics["domain_loss"] = running_domain_loss / len(loader)
            metrics["domain_acc"]  = domain_correct / max(domain_total, 1)
            metrics["lambda"]      = current_lambda

        if self._uses_multitask and train:
            metrics["cluster_loss"] = running_cluster_loss / len(loader)

        # Per-cluster MAE (diagnostic)
        cluster_metrics: Dict[str, float] = {}
        for cid in range(5):
            mask = np.array(all_clusters) == cid
            if mask.sum() == 0:
                continue
            cp = np.array(all_preds)[mask]
            ct = np.array(all_targets)[mask]
            # Slice city stats to match the cluster subset
            cm = _c_means[mask]
            cs = _c_stds[mask]
            cp = torch.expm1(torch.from_numpy(cp) * torch.from_numpy(cs) + torch.from_numpy(cm)).numpy()
            ct = torch.expm1(torch.from_numpy(ct) * torch.from_numpy(cs) + torch.from_numpy(cm)).numpy()
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
        if self._uses_dann:
            history.update({k: [] for k in
                            ["train_domain_loss", "train_domain_acc", "lambda"]})
        if self._uses_multitask:
            history["train_cluster_loss"] = []

        for epoch in range(tc.num_epochs):
            t0 = time.time()
            ep = epoch + 1

            # Unfreeze backbone after warm-up
            if self._has_backbone and self._warmup_epochs > 0 and epoch == self._warmup_epochs:
                for p in self.model.backbone.parameters():
                    p.requires_grad = True
                print(f"\n  ── Warm-up done: backbone unfrozen "
                      f"(backbone lr={tc.learning_rate*0.01:.2e}) ──")

            train_m, _         = self._run_epoch(self.train_loader, train=True,
                                                  desc=f"Ep {ep}/{tc.num_epochs} [train]",
                                                  epoch=epoch)
            val_m,   cluster_m = self._run_epoch(self.val_loader,   train=False,
                                                  desc=f"Ep {ep}/{tc.num_epochs} [val]  ",
                                                  epoch=epoch)

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_m["loss"])
            new_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            aux_str = ""
            if self._uses_dann:
                aux_str = (
                    f"\n  DANN   λ={train_m.get('lambda', 0):.3f}  "
                    f"dom_loss={train_m.get('domain_loss', 0):.4f}  "
                    f"dom_acc={train_m.get('domain_acc', 0):.3f}"
                )
            elif self._uses_multitask:
                aux_str = (
                    f"\n  MultiTask  cluster_loss={train_m.get('cluster_loss', 0):.4f}"
                )
            print(
                f"\nEpoch {ep}/{tc.num_epochs}  "
                f"({elapsed:.1f}s  lr={new_lr:.2e})\n"
                f"  train  loss={train_m['loss']:.4f}  "
                f"MAE={train_m['mae']:.1f}  R²={train_m['r2']:.4f}\n"
                f"  val    loss={val_m['loss']:.4f}  "
                f"MAE={val_m['mae']:.1f}  R²={val_m['r2']:.4f}"
                + aux_str
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
                log_dict = {
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
                }
                if self._uses_dann:
                    log_dict.update({
                        "dann/lambda":      train_m.get("lambda", 0),
                        "dann/domain_loss": train_m.get("domain_loss", 0),
                        "dann/domain_acc":  train_m.get("domain_acc", 0),
                    })
                if self._uses_multitask:
                    log_dict["multitask/cluster_loss"] = train_m.get("cluster_loss", 0)
                wandb.log(log_dict)

            # History
            for key in ("loss", "mae", "r2"):
                history[f"train_{key}"].append(float(train_m[key]))
                history[f"val_{key}"].append(float(val_m[key]))
            if self._uses_dann:
                history["train_domain_loss"].append(float(train_m.get("domain_loss", 0)))
                history["train_domain_acc"].append(float(train_m.get("domain_acc", 0)))
                history["lambda"].append(float(train_m.get("lambda", 0)))
            if self._uses_multitask:
                history["train_cluster_loss"].append(float(train_m.get("cluster_loss", 0)))

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
