"""
Tabular feature attribution for dual-branch population prediction models.

Uses Gradient × Input to estimate how much each of the 15 OSM features
contributes to each prediction on the test set.  Gradient × Input is the
element-wise product of the input feature value and the gradient of the
prediction w.r.t. that feature — a fast, parameter-free attribution method.

Produces:
  shap/feature_importance.png     — mean |attribution| per feature (bar chart)
  shap/attribution_direction.png  — signed mean attribution (direction of effect)
  shap/beeswarm.png               — per-sample attribution scatter (SHAP-style)
  shap/cluster_heatmap.png        — per-cluster mean attribution heatmap
  shap/attributions.npz           — raw attribution matrix for further analysis

Only meaningful for models with a tabular branch (dual, crossattn, film,
dann, multitask).

Usage:
    python visualize_shap.py \\
        --cache /workspace/PopulationDataset/cache \\
        --run   dual_convnext_tiny \\
        --model dual \\
        --backbone convnext_tiny \\
        --json  /workspace/PopulationDataset/final_clustered_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",       required=True)
    p.add_argument("--run",         required=True)
    p.add_argument("--model",       required=True,
                   choices=["dual", "crossattn", "film", "dann", "multitask"])
    p.add_argument("--backbone",    default="convnext_tiny")
    p.add_argument("--json",        default=None,
                   help="Path to final_clustered_samples.json (for feature names)")
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size",  type=int, default=32)
    return p.parse_args()


# ── Feature names ─────────────────────────────────────────────────────────────

def load_feature_names(json_path: str | None, n: int = 15) -> list[str]:
    """Try to extract OSM feature names from the original JSON."""
    if json_path and Path(json_path).exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            for sample in data:
                feats = sample.get("osm_features")
                if isinstance(feats, dict) and len(feats) == n:
                    return list(feats.keys())
                if isinstance(feats, list) and len(feats) == n:
                    break   # list format — no names available
        except Exception:
            pass
    # Fallback: generic names
    return [f"osm_feat_{i:02d}" for i in range(n)]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type, backbone, ckpt_path, device):
    from configs import (DualBranchConfig, CrossAttnConfig,
                         FiLMConfig, DANNConfig, MultiTaskConfig)
    from models import build_model

    cfg_map = {
        "dual":      DualBranchConfig(backbone=backbone),
        "crossattn": CrossAttnConfig(backbone=backbone),
        "film":      FiLMConfig(backbone=backbone),
        "dann":      DANNConfig(backbone=backbone),
        "multitask": MultiTaskConfig(backbone=backbone),
    }
    model = build_model(cfg_map[model_type], device=device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Attribution computation ───────────────────────────────────────────────────

def compute_attributions(model, loader, model_type, device, denorm):
    """
    Gradient × Input attribution for the tabular branch.

    For each sample:
      attr_i = (d pred / d tab_i) * tab_i

    Returns:
        attrs    (N, 15) float32 — attribution per feature per sample
        tabular  (N, 15) float32 — raw tabular input values
        preds    (N,)    float32 — denormalized predictions
        targets  (N,)    float32 — denormalized targets
        clusters (N,)    int
    """
    all_attrs, all_tab, all_pred, all_tgt, all_clust = [], [], [], [], []

    for batch in tqdm(loader, desc="Computing attributions"):
        images  = batch["image"].to(device)
        tabular = batch["tabular"].to(device).detach().requires_grad_(True)
        targets = batch["target"].squeeze()

        if model_type == "dann":
            preds, _ = model(images, tabular, 0.0)
        else:
            preds = model(images, tabular)

        preds = preds.squeeze()
        model.zero_grad()
        preds.sum().backward()

        attr = (tabular.grad * tabular).detach().cpu().float()

        all_attrs.append(attr.numpy())
        all_tab.append(tabular.detach().cpu().numpy())
        all_pred.extend(denorm(preds.detach().cpu()).tolist())
        all_tgt.extend(denorm(targets).tolist())
        all_clust.extend(batch["metadata"]["cluster"].numpy().tolist())

    return (
        np.vstack(all_attrs),
        np.vstack(all_tab),
        np.array(all_pred, dtype=np.float32),
        np.array(all_tgt,  dtype=np.float32),
        np.array(all_clust, dtype=int),
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_importance(attrs, feat_names, out_path):
    """Horizontal bar chart of mean |attribution| per feature."""
    mean_abs = np.abs(attrs).mean(axis=0)   # (15,)
    order    = np.argsort(mean_abs)          # ascending → bottom is most important

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(feat_names) + 2))
    bars = ax.barh(
        [feat_names[i] for i in order],
        mean_abs[order],
        color=plt.cm.Blues(0.6 + 0.4 * mean_abs[order] / (mean_abs.max() + 1e-8)),
    )
    ax.set_xlabel("Mean |Gradient × Input|", fontsize=11)
    ax.set_title("Tabular Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_direction(attrs, feat_names, out_path):
    """Signed mean attribution — shows direction of each feature's effect."""
    mean_attr = attrs.mean(axis=0)   # (15,)
    order     = np.argsort(mean_attr)

    colors = ["#d73027" if v > 0 else "#4575b4" for v in mean_attr[order]]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(feat_names) + 2))
    ax.barh([feat_names[i] for i in order], mean_attr[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean (Gradient × Input)", fontsize=11)
    ax.set_title("Feature Attribution Direction\n"
                 "(red = increases prediction, blue = decreases prediction)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_beeswarm(attrs, tab_vals, feat_names, out_path, n_feat=10):
    """
    SHAP-style beeswarm: for each top feature, scatter of per-sample
    attribution (x) vs sample index (y), coloured by feature value.
    """
    mean_abs = np.abs(attrs).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-n_feat:][::-1]   # top N by importance

    fig, axes = plt.subplots(n_feat, 1, figsize=(9, 2.2 * n_feat), sharex=False)
    if n_feat == 1:
        axes = [axes]

    norm = mcolors.Normalize()

    for ax, fi in zip(axes, top_idx):
        feat_vals = tab_vals[:, fi]
        attr_vals = attrs[:, fi]
        norm.autoscale(feat_vals)
        colors = plt.cm.RdBu_r(norm(feat_vals))

        # Jitter y-axis for visibility
        rng = np.random.default_rng(fi)
        y   = rng.uniform(-0.3, 0.3, size=len(attr_vals))
        sc  = ax.scatter(attr_vals, y, c=colors, alpha=0.5, s=10, linewidths=0)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_ylabel(feat_names[fi], fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=4)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("feat val", fontsize=7)

    axes[-1].set_xlabel("Attribution (Gradient × Input)", fontsize=10)
    plt.suptitle(f"Feature Attribution Distribution (top {n_feat})",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_cluster_heatmap(attrs, clusters, feat_names, out_path):
    """Per-cluster mean attribution heatmap."""
    n_clusters = int(clusters.max()) + 1
    cluster_attrs = np.zeros((n_clusters, attrs.shape[1]))
    for c in range(n_clusters):
        mask = clusters == c
        if mask.sum() > 0:
            cluster_attrs[c] = attrs[mask].mean(axis=0)

    # Sort features by overall variance across clusters
    order = np.argsort(cluster_attrs.std(axis=0))[::-1]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_names) * 0.7), 4))
    im = ax.imshow(cluster_attrs[:, order], aspect="auto", cmap="RdBu_r",
                   norm=mcolors.TwoSlopeNorm(vcenter=0))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([feat_names[i] for i in order],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {c}" for c in range(n_clusters)])
    plt.colorbar(im, ax=ax, label="Mean attribution")
    ax.set_title("Per-Cluster Feature Attribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path("outputs") / args.run / "shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Run: {args.run}")

    ckpt = Path("outputs") / args.run / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = load_model(args.model, args.backbone, ckpt, device)

    from data.dataset import CachedDataset
    test_ds = CachedDataset(args.cache, "test", use_tabular=True)
    denorm  = test_ds.denormalize_target
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    feat_names = load_feature_names(args.json, n=15)
    print(f"Feature names: {feat_names}")

    attrs, tab_vals, preds, targets, clusters = compute_attributions(
        model, loader, args.model, device, denorm
    )

    # Save raw data
    np.savez(out_dir / "attributions.npz",
             attrs=attrs, tabular=tab_vals,
             preds=preds, targets=targets, clusters=clusters)
    print(f"  Saved attributions.npz  (shape: {attrs.shape})")

    # Plots
    plot_importance(attrs, feat_names,  out_dir / "feature_importance.png")
    plot_direction(attrs,  feat_names,  out_dir / "attribution_direction.png")
    plot_beeswarm(attrs, tab_vals, feat_names,
                  out_dir / "beeswarm.png", n_feat=min(10, len(feat_names)))
    plot_cluster_heatmap(attrs, clusters, feat_names,
                         out_dir / "cluster_heatmap.png")

    # Print top-5 summary
    mean_abs = np.abs(attrs).mean(axis=0)
    top5 = np.argsort(mean_abs)[-5:][::-1]
    print("\nTop-5 most influential features:")
    for rank, fi in enumerate(top5, 1):
        print(f"  {rank}. {feat_names[fi]:30s}  mean|attr|={mean_abs[fi]:.5f}")

    print(f"\nSHAP analysis done → {out_dir}/")


if __name__ == "__main__":
    main()
