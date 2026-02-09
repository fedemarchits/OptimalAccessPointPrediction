"""
TRAINING SCRIPT - Baseline Population Predictor

Complete script to train the baseline model with:
- Training & Validation loops
- Metrics: MAE, RMSE, R²
- Early stopping
- Checkpointing
- TensorBoard logging
- Progress tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json

from geospatial_dataset_multichannel import get_dataloaders
from baseline_model_multichannel import get_model


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute all regression metrics

    Args:
        predictions: Array (N,) of predictions
        targets: Array (N,) of ground truth values

    Returns:
        dict with MAE, RMSE, R², MAPE
    """
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions - targets))

    # RMSE (Root Mean Squared Error)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # R² (Coefficient of Determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    model_type: str = 'baseline'
) -> Dict[str, float]:
    """
    Train one epoch

    Returns:
        dict with loss and metrics
    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Data
        images = batch['image'].to(device)
        targets = batch['target'].to(device).squeeze()  # (B,)

        # Forward
        optimizer.zero_grad()
        
        if model_type == 'dual_branch':
            # Dual-branch model needs both images and tabular features
            tabular = batch['tabular'].to(device)
            predictions = model(images, tabular).squeeze()  # (B,)
        else:
            # Baseline model only needs images
            predictions = model(images).squeeze()  # (B,)

        # Loss
        loss = criterion(predictions, targets)

        # Backward
        loss.backward()

        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        # Stats
        running_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        # Progress bar
        pbar.set_postfix({'loss': loss.item()})

        # TensorBoard (every 100 batches)
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

    # Epoch metrics
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics['loss'] = avg_loss

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    model_type: str = 'baseline'
) -> Dict[str, float]:
    """
    Validation

    Returns:
        dict with loss and metrics
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_cities = []
    all_clusters = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ")

    for batch in pbar:
        # Data
        images = batch['image'].to(device)
        targets = batch['target'].to(device).squeeze()

        # Forward
        if model_type == 'dual_branch':
            tabular = batch['tabular'].to(device)
            predictions = model(images, tabular).squeeze()
        else:
            predictions = model(images).squeeze()

        # Loss
        loss = criterion(predictions, targets)

        # Stats
        running_loss += loss.item()
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_cities.extend(batch['metadata']['city'])
        all_clusters.extend(batch['metadata']['cluster'].numpy())

        # Progress bar
        pbar.set_postfix({'loss': loss.item()})

    # Metrics
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics['loss'] = avg_loss

    # Cluster-level metrics (diagnostic info)
    cluster_metrics = {}
    for cluster_id in range(5):
        mask = np.array(all_clusters) == cluster_id
        if mask.sum() > 0:
            cluster_preds = np.array(all_preds)[mask]
            cluster_targets = np.array(all_targets)[mask]
            cluster_mae = np.mean(np.abs(cluster_preds - cluster_targets))
            cluster_metrics[f'cluster_{cluster_id}_mae'] = cluster_mae

    return metrics, cluster_metrics


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum required improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = 'cuda'
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, Dict]:
    """
    Load checkpoint

    Returns:
        (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})

    return model, optimizer, epoch, metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    # Data
    json_file: str,
    base_dir: str,  # Base directory containing satellite_images/, dem_height/, segmentation_land_use/

    # Model
    model_type: str = 'baseline',
    pretrained: bool = True,
    freeze_backbone: bool = False,

    # Training
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,

    # Early stopping
    patience: int = 15,

    # Checkpointing
    checkpoint_dir: str = 'checkpoints',
    save_every: int = 5,

    # Logging
    log_dir: str = 'runs/baseline',

    # System
    num_workers: int = 4,
    device: str = None,
    resume_from: Optional[str] = None
):
    """
    Main training function
    """

    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING BASELINE MODEL")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {patience}")

    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================

    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    use_tabular = (model_type == 'dual_branch')

    train_loader, val_loader, test_loader = get_dataloaders(
        json_file=json_file,
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tabular=use_tabular
    )

    print("\nDataLoaders ready:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # ========================================================================
    # 2. MODEL
    # ========================================================================

    print("\n" + "=" * 80)
    print("Model Creation")
    print("=" * 80)

    model = get_model(
        model_type=model_type,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device
    )

    # ========================================================================
    # 3. LOSS & OPTIMIZER
    # ========================================================================

    criterion = nn.MSELoss()  # Mean Squared Error for regression

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    print(f"\nOptimizer: Adam (lr={learning_rate}, wd={weight_decay})")
    print("Loss: MSE")
    print("Scheduler: ReduceLROnPlateau")

    # ========================================================================
    # 4. RESUME FROM CHECKPOINT (OPTIONAL)
    # ========================================================================

    start_epoch = 0

    if resume_from is not None:
        print(f"\nResuming from checkpoint: {resume_from}")
        model, optimizer, start_epoch, _ = load_checkpoint(
            Path(resume_from),
            model,
            optimizer,
            device
        )
        print(f"Checkpoint loaded (epoch {start_epoch})")
        start_epoch += 1

    # ========================================================================
    # 5. LOGGING
    # ========================================================================

    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard log directory: {log_dir}")
    print(f"Launch with: tensorboard --logdir={log_dir}")

    # ========================================================================
    # 6. EARLY STOPPING
    # ========================================================================

    early_stopping = EarlyStopping(patience=patience)

    # ========================================================================
    # 7. TRAINING LOOP
    # ========================================================================

    print("\n" + "=" * 80)
    print("Training Loop")
    print("=" * 80)

    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_mae': [],
        'train_r2': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': []
    }

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, writer, model_type=model_type
        )

        val_metrics, cluster_metrics = validate(
            model, val_loader, criterion,
            device, epoch, writer, model_type=model_type
        )

        scheduler.step(val_metrics['loss'])

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.2f} | R²: {train_metrics['r2']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.2f} | R²: {val_metrics['r2']:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)

        writer.add_scalars('MAE', {
            'train': train_metrics['mae'],
            'val': val_metrics['mae']
        }, epoch)

        writer.add_scalars('R2', {
            'train': train_metrics['r2'],
            'val': val_metrics['r2']
        }, epoch)

        for cluster_name, cluster_mae in cluster_metrics.items():
            writer.add_scalar(f'Val/ClusterMAE/{cluster_name}', cluster_mae, epoch)

        history['train_loss'].append(train_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / 'best_model.pth'
            )
            print(f"  Best model saved (Val Loss: {best_val_loss:.4f})")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )
            print("  Checkpoint saved")

        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
            break

    # ========================================================================
    # 8. FINAL EVALUATION ON TEST SET
    # ========================================================================

    print("\n" + "=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)

    model, _, _, _ = load_checkpoint(
        checkpoint_dir / 'best_model.pth',
        model,
        device=device
    )
    print(f"Best model loaded (epoch {best_epoch})")

    test_metrics, test_cluster_metrics = validate(
        model, test_loader, criterion,
        device, epoch=-1, writer=None, model_type=model_type
    )

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f} people")
    print(f"  RMSE: {test_metrics['rmse']:.2f} people")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    print("\nTest Performance per Cluster:")
    for cluster_name, cluster_mae in test_cluster_metrics.items():
        print(f"  {cluster_name}: MAE = {cluster_mae:.2f}")

    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved: {checkpoint_dir / 'training_history.json'}")

    writer.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")