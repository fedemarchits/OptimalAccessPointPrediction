"""
Per-country error analysis for population prediction models.

Runs inference on the full test set, groups predictions by country, and
produces two visualizations:

  country_errors/country_mae_bar.png    — horizontal bar chart, sorted by MAE
  country_errors/country_map.png        — choropleth map over Europe
                                          (requires geopandas; skipped otherwise)
  country_errors/country_errors.json    — raw per-country metrics

Also produces a cross-model comparison if run as part of run_all.py.

Usage:
    python visualize_country_errors.py \\
        --cache   /workspace/PopulationDataset/cache \\
        --run     dual_convnext_tiny \\
        --model   dual \\
        --backbone convnext_tiny
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import geopandas as gpd
    _GEO_OK = True
except ImportError:
    _GEO_OK = False


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",       required=True)
    p.add_argument("--run",         required=True)
    p.add_argument("--model",       required=True,
                   choices=["single", "dual", "crossattn", "film", "dann", "multitask"])
    p.add_argument("--backbone",    default="convnext_tiny")
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type, backbone, ckpt_path, device):
    from configs import (SingleBranchConfig, DualBranchConfig, CrossAttnConfig,
                         FiLMConfig, DANNConfig, MultiTaskConfig)
    from models import build_model

    cfg_map = {
        "single":    SingleBranchConfig(backbone=backbone),
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


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, loader, model_type, device, denorm):
    """Returns per-country lists of (pred, target)."""
    country_preds   = defaultdict(list)
    country_targets = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs  = batch["image"].to(device)
            tab   = batch["tabular"].to(device)
            tgts  = batch["target"].squeeze()
            countries = batch["metadata"]["country"]

            if model_type == "single":
                preds = model(imgs)
            elif model_type == "dann":
                preds, _ = model(imgs, tab, 0.0)
            else:
                preds = model(imgs, tab)

            preds_d = denorm(preds.squeeze().cpu())
            tgts_d  = denorm(tgts.cpu())

            for i, country in enumerate(countries):
                country_preds[country].append(float(preds_d[i]))
                country_targets[country].append(float(tgts_d[i]))

    return country_preds, country_targets


def compute_country_metrics(country_preds, country_targets):
    """Compute MAE, RMSE, R², N per country."""
    metrics = {}
    for country in sorted(country_preds):
        p = np.array(country_preds[country])
        t = np.array(country_targets[country])
        n = len(p)
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        metrics[country] = {"mae": mae, "rmse": rmse, "r2": r2, "n": n}
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bar(metrics, out_path, run_name):
    countries = sorted(metrics, key=lambda c: metrics[c]["mae"])
    maes  = [metrics[c]["mae"] for c in countries]
    ns    = [metrics[c]["n"]   for c in countries]

    overall_mae = np.mean([metrics[c]["mae"] for c in metrics])

    fig, ax = plt.subplots(figsize=(9, max(5, 0.55 * len(countries) + 2)))

    norm   = plt.Normalize(min(maes), max(maes))
    colors = plt.cm.RdYlGn_r(norm(maes))

    bars = ax.barh(countries, maes, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate with sample count
    for bar, n in zip(bars, ns):
        ax.text(bar.get_width() + max(maes) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", ha="left", fontsize=8, color="gray")

    ax.axvline(overall_mae, color="#333", linestyle="--", linewidth=1.2,
               label=f"Overall MAE = {overall_mae:.0f}")
    ax.set_xlabel("Mean Absolute Error (population count)", fontsize=11)
    ax.set_title(f"Per-Country MAE — {run_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="MAE", fraction=0.03, pad=0.12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_map(metrics, out_path, run_name):
    """Choropleth map of per-country MAE over Europe."""
    if not _GEO_OK:
        print("  geopandas not available — skipping choropleth map")
        return

    try:
        # Try naturalearth_lowres (bundled with geopandas)
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        europe = world[world["continent"] == "Europe"].copy()

        # Align country names (dataset uses "city, Country" format)
        # Common mismatches between dataset names and naturalearth names
        name_map = {
            "Czech Republic": "Czechia",
            "Slovak Republic": "Slovakia",
            "United Kingdom": "United Kingdom",
        }
        mae_series = {name_map.get(k, k): v["mae"] for k, v in metrics.items()}

        europe["mae"] = europe["name"].map(mae_series)

        fig, ax = plt.subplots(figsize=(12, 7))
        europe[europe["mae"].isna()].plot(
            ax=ax, color="#dddddd", edgecolor="white", linewidth=0.4
        )
        europe[europe["mae"].notna()].plot(
            ax=ax, column="mae", cmap="RdYlGn_r",
            edgecolor="white", linewidth=0.4, legend=True,
            legend_kwds={"label": "MAE", "shrink": 0.6},
        )

        # Clip to Europe bounding box
        ax.set_xlim(-15, 35)
        ax.set_ylim(33, 72)
        ax.set_title(f"Per-Country MAE — {run_name}", fontsize=13, fontweight="bold")
        ax.axis("off")

        # Annotate each country with its MAE
        for _, row in europe[europe["mae"].notna()].iterrows():
            centroid = row.geometry.centroid
            if -15 <= centroid.x <= 35 and 33 <= centroid.y <= 72:
                ax.text(centroid.x, centroid.y, f"{row['mae']:.0f}",
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="black", alpha=0.85)

        plt.tight_layout()
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")

    except Exception as e:
        print(f"  Choropleth map failed ({e}) — bar chart only")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path("outputs") / args.run / "country_errors")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Run: {args.run}")

    ckpt = Path("outputs") / args.run / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = load_model(args.model, args.backbone, ckpt, device)

    from data.dataset import CachedDataset
    uses_tabular = args.model != "single"
    test_ds = CachedDataset(args.cache, "test", use_tabular=uses_tabular)
    denorm  = test_ds.denormalize_target
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    country_preds, country_targets = run_inference(
        model, loader, args.model, device, denorm
    )
    metrics = compute_country_metrics(country_preds, country_targets)

    # Save raw metrics
    with open(out_dir / "country_errors.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved country_errors.json")

    # Print summary table
    print(f"\n{'Country':<25} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'N':>5}")
    print("-" * 55)
    for c, m in sorted(metrics.items(), key=lambda x: x[1]["mae"]):
        print(f"{c:<25} {m['mae']:>8.0f} {m['rmse']:>8.0f} {m['r2']:>7.3f} {m['n']:>5}")

    plot_bar(metrics, out_dir / "country_mae_bar.png", args.run)
    plot_map(metrics, out_dir / "country_map.png",     args.run)

    print(f"\nCountry error analysis done → {out_dir}/")


if __name__ == "__main__":
    main()
