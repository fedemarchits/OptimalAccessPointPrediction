"""
UMAP feature-space visualization for domain invariance analysis.

Extracts backbone features from val+test samples, reduces to 2D with UMAP,
and produces three scatter plots:
  1. Coloured by country   — should dissolve after DANN training
  2. Coloured by population — should remain structured (task signal preserved)
  3. Coloured by cluster   — diagnostic: urban-fabric separation

Also computes:
  - Silhouette score  (country labels): high = country clusters are tight (bad for DANN)
  - Adjusted Rand Index vs country (KMeans k=n_countries): same interpretation

Usage:
    python visualize_umap.py \\
        --cache  /workspace/PopulationDataset/cache \\
        --run    dual_efficientnet_b3 \\
        --model  dual \\
        --backbone efficientnet_b3

    # DANN model
    python visualize_umap.py \\
        --cache  /workspace/PopulationDataset/cache \\
        --run    dann_efficientnet_b3 \\
        --model  dann \\
        --backbone efficientnet_b3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UMAP feature-space visualization")
    p.add_argument("--cache",      required=True,  help="Path to preprocessed cache dir")
    p.add_argument("--run",        required=True,  help="Run name (used to find checkpoint)")
    p.add_argument("--model",      required=True,
                   choices=["single", "dual", "crossattn", "film", "dann", "multitask"],
                   help="Model type")
    p.add_argument("--backbone",   default="efficientnet_b3")
    p.add_argument("--output-dir", default=None,   help="Where to save plots (default: outputs/{run}/umap)")
    p.add_argument("--splits",     nargs="+", default=["val", "test"],
                   help="Splits to extract features from (default: val test)")
    p.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--min-dist",    type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type: str, backbone: str, checkpoint_path: Path, device: str):
    from models.backbones.base import build_backbone

    bb = build_backbone(backbone, pretrained=False, freeze=False)

    if model_type == "single":
        from models.single_branch import SingleBranchModel
        model = SingleBranchModel(bb)
    elif model_type == "dual":
        from models.dual_branch import DualBranchModel
        model = DualBranchModel(bb)
    elif model_type == "crossattn":
        from models.dual_branch import CrossAttnDualBranch
        model = CrossAttnDualBranch(bb)
    elif model_type == "film":
        from models.dual_branch import FiLMDualBranch
        model = FiLMDualBranch(bb)
    elif model_type == "dann":
        from models.dual_branch import DANNDualBranch
        # n_domains will be inferred from checkpoint shape
        model = DANNDualBranch(bb, n_domains=12)
    elif model_type == "multitask":
        from models.dual_branch import MultiTaskDualBranch
        model = MultiTaskDualBranch(bb)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Handle n_domains mismatch for DANN
    if model_type == "dann" and "domain_classifier.4.weight" in state:
        n_domains_ckpt = state["domain_classifier.4.weight"].shape[0]
        if n_domains_ckpt != 12:
            model = DANNDualBranch(bb, n_domains=n_domains_ckpt)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract backbone features (B, feat_dim) for all samples in loader.

    Returns arrays: features, targets (log-pop), countries (str), clusters (int).
    """
    all_feats    = []
    all_targets  = []
    all_countries = []
    all_clusters  = []

    for batch in tqdm(loader, desc="Extracting features", leave=False):
        images  = batch["image"].to(device)
        tabular = batch["tabular"].to(device)

        # All model types expose model.backbone(image) → (B, feat_dim)
        feats = model.backbone(images)   # (B, feat_dim)
        all_feats.append(feats.cpu().numpy())

        # Targets — stored as log1p-normalised population in the cache
        targets = batch["target"].squeeze().cpu().numpy()
        all_targets.extend(np.atleast_1d(targets))

        # Country and cluster from metadata
        meta = batch["metadata"]
        all_countries.extend(meta["country"])
        all_clusters.extend(meta["cluster"].numpy().tolist())

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    targets  = np.array(all_targets, dtype=np.float32)
    clusters = np.array(all_clusters, dtype=np.int32)
    return features, targets, np.array(all_countries), clusters


# ── UMAP ─────────────────────────────────────────────────────────────────────

def run_umap(features: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42, verbose=False)
    print(f"Running UMAP on {features.shape[0]} samples × {features.shape[1]} dims…")
    return reducer.fit_transform(features)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(features: np.ndarray, country_labels: np.ndarray,
                    n_countries: int) -> dict:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    int_labels = le.fit_transform(country_labels)

    sil = silhouette_score(features, int_labels, sample_size=min(5000, len(features)),
                           random_state=42)

    km = KMeans(n_clusters=n_countries, random_state=42, n_init=10).fit(features)
    ari = adjusted_rand_score(int_labels, km.labels_)

    return {"silhouette": float(sil), "ari": float(ari)}


def per_country_silhouette(embedding: np.ndarray, country_labels: np.ndarray) -> dict:
    """Compute silhouette score in the 2D UMAP space per country."""
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    int_labels = le.fit_transform(country_labels)
    if len(np.unique(int_labels)) < 2:
        return {}
    scores = silhouette_samples(embedding, int_labels)
    return {country: float(scores[country_labels == country].mean())
            for country in np.unique(country_labels)}


# ── Plotting ──────────────────────────────────────────────────────────────────

_CMAP_QUAL = plt.cm.get_cmap("tab20")


def _country_colours(countries: np.ndarray):
    unique = sorted(np.unique(countries))
    palette = {c: _CMAP_QUAL(i / max(len(unique) - 1, 1)) for i, c in enumerate(unique)}
    colours = np.array([palette[c] for c in countries])
    return colours, palette, unique


def plot_by_country(emb, countries, out_path, title="Feature space — coloured by country"):
    fig, ax = plt.subplots(figsize=(10, 8))
    colours, palette, unique = _country_colours(countries)
    ax.scatter(emb[:, 0], emb[:, 1], c=colours, s=6, alpha=0.6, linewidths=0)
    patches = [mpatches.Patch(color=palette[c], label=c) for c in unique]
    ax.legend(handles=patches, fontsize=7, loc="best", ncol=2,
              framealpha=0.7, markerscale=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_continuous(emb, values, out_path, label="log-population",
                       cmap="plasma", title=None):
    title = title or f"Feature space — coloured by {label}"
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap=cmap, s=6, alpha=0.7,
                    linewidths=0)
    plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_silhouette_bar(per_country: dict, out_path,
                        title="Per-country silhouette score (UMAP space)"):
    if not per_country:
        return
    countries = sorted(per_country.keys())
    scores    = [per_country[c] for c in countries]
    colours   = ["#d73027" if s > 0 else "#4575b4" for s in scores]

    fig, ax = plt.subplots(figsize=(max(8, len(countries) * 0.7), 4))
    ax.bar(range(len(countries)), scores, color=colours, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Silhouette score")
    ax.set_title(title + "\n(red = country cluster clearly visible, blue = mixed — blue is better for DANN)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve paths
    ckpt_path = Path("outputs") / args.run / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / args.run / "umap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Run      : {args.run}")
    print(f"  Model    : {args.model} / {args.backbone}")
    print(f"  Splits   : {args.splits}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.model, args.backbone, ckpt_path, device)

    # ── Build dataloaders for the requested splits ─────────────────────────
    from data.dataset import CachedDataset

    datasets = []
    for split in args.splits:
        ds = CachedDataset(args.cache, split=split, crop_size=224,
                           use_tabular=True, augment=False)
        datasets.append(ds)

    # Concatenate splits
    from torch.utils.data import ConcatDataset
    combined_ds = ConcatDataset(datasets)
    loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    # n_countries from first dataset
    n_countries = datasets[0].n_countries

    # ── Extract features ──────────────────────────────────────────────────────
    features, targets, countries, clusters = extract_features(model, loader, device)
    print(f"\nExtracted {features.shape[0]} feature vectors of dim {features.shape[1]}")
    print(f"Countries: {sorted(np.unique(countries))}")

    # ── High-dim metrics (on raw features, not UMAP) ──────────────────────────
    print("\nComputing domain-separation metrics on raw features…")
    metrics = compute_metrics(features, countries, n_countries)
    print(f"  Silhouette score (country, raw features): {metrics['silhouette']:.4f}")
    print(f"  Adjusted Rand Index (KMeans vs country):  {metrics['ari']:.4f}")
    print(f"  Interpretation: closer to 0 = better domain invariance")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    emb = run_umap(features, args.n_neighbors, args.min_dist)
    np.save(out_dir / "umap_embedding.npy", emb)

    # ── Per-country silhouette in UMAP space ──────────────────────────────────
    per_sil = per_country_silhouette(emb, countries)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")

    plot_by_country(
        emb, countries,
        out_dir / "umap_by_country.png",
        title=f"{args.run} — feature space by country\n"
              f"(Silhouette={metrics['silhouette']:.3f}, ARI={metrics['ari']:.3f})",
    )

    plot_by_continuous(
        emb, targets,
        out_dir / "umap_by_population.png",
        label="log1p population",
        cmap="plasma",
        title=f"{args.run} — feature space by log-population\n"
              f"(should stay structured even after DANN)",
    )

    cluster_labels = [f"cluster {c}" for c in clusters]
    plot_by_continuous(
        emb, clusters.astype(float),
        out_dir / "umap_by_cluster.png",
        label="K-Means cluster (0–4)",
        cmap="tab10",
        title=f"{args.run} — feature space by urban cluster",
    )

    plot_silhouette_bar(per_sil, out_dir / "silhouette_by_country.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Silhouette (country, raw feats) : {metrics['silhouette']:.4f}")
    print(f"  ARI (KMeans vs country)          : {metrics['ari']:.4f}")
    print(f"  Plots saved to: {out_dir}")
    print(f"{'='*60}")
    print(
        "\nInterpretation guide:\n"
        "  Silhouette → 0   and   ARI → 0 :  DANN is working — features are country-agnostic\n"
        "  Silhouette >> 0  and   ARI >> 0 :  features still encode country identity\n"
        "  (compare these numbers between the baseline and DANN runs)"
    )


if __name__ == "__main__":
    main()
