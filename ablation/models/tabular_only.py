"""
Tabular-only MLP model (no image backbone).

Ablation baseline that uses only the 15 OSM tabular features to predict
population, with no satellite imagery at all.  Answers the question:
"How much of the signal comes from OSM features alone?"
"""

import torch
import torch.nn as nn


class TabularOnlyModel(nn.Module):
    """
    Deep MLP operating solely on the 15 OSM tabular features.
    No image branch, no backbone — pure tabular prediction.
    """

    uses_tabular_only: bool = True

    def __init__(self, tabular_dim: int = 15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.mlp(tabular)
