"""Regression metrics with optional denormalisation."""

from __future__ import annotations
from typing import Callable, Dict, Optional

import numpy as np
import torch


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    denormalize_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² and MAPE.

    If denormalize_fn is provided, both arrays are transformed back to
    the original scale before metric computation so that numbers are
    interpretable (e.g. MAE in actual population counts, not log space).
    """
    if denormalize_fn is not None:
        predictions = denormalize_fn(torch.from_numpy(predictions)).numpy()
        targets     = denormalize_fn(torch.from_numpy(targets)).numpy()

    mae  = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    mask = targets != 0
    mape = float(
        np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    ) if mask.sum() > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
