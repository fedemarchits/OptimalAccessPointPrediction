"""
Population Prediction Visualizer

Generates three complementary visualizations for a given city:

  1. Ground-truth overlay
       RGB satellite image + colour-coded sample points
       (colour and size = population within 15-min walk)

  2. Dense prediction heatmap
       Model slides a 224×224 window across the entire city TIF at a
       configurable stride and builds a smooth population map overlay.
       Tabular features are set to the city-average from the JSON.

  3. Attention map  (CrossAttnDualBranch only)
       For a chosen sample point, shows the 7×7 cross-attention weights
       upsampled and overlaid on the 224×224 crop.

Usage (on the Vast.ai instance):
    cd /workspace/OptimalAccessPointPrediction/ablation

    python visualize.py \\
        --city      "Bologna, Italy" \\
        --checkpoint outputs/crossattn_efficientnet_b3/best_model.pth \\
        --model     crossattn \\
        --json      /workspace/PopulationDataset/final_clustered_samples.json \\
        --base      /workspace/PopulationDataset \\
        --cache     /workspace/PopulationDataset/cache \\
        --out       /workspace/visualizations

    --model  choices: single | dual | crossattn
    --cache  path to preprocessed cache dir (required for correct tabular z-score)
    --stride controls sliding-window density (default 56, lower = denser but slower)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")          # headless — no display needed on server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ──────────────────────────────────────────────────────────────
_DEM_MIN, _DEM_MAX   = -50.0, 3_000.0
N_TABULAR            = 15
TARGET_CLAMP_MAX     = 80_000.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _clean_key(s: str) -> str:
    s = str(s).replace(".tif", "").replace(".tiff", "")
    for suf in ["_LandUse", "LandUse", "_DEM", "DEM", "_10m", "10m"]:
        s = s.replace(suf, "")
    s = s.lower()
    for ch, rep in {"ø":"o","å":"a","æ":"ae","ö":"o","ü":"u","ä":"a","é":"e","è":"e"}.items():
        s = s.replace(ch, rep)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum())


def _find_city_files(base_dir: Path, city: str):
    key = _clean_key(city)
    rgb_dir = base_dir / "satellite_images"
    dem_dir = base_dir / "dem_height"
    lu_dir  = base_dir / "segmentation_land_use"
    rgb = next((f for f in rgb_dir.glob("*.tif") if _clean_key(f.name) == key), None)
    dem = next((f for f in dem_dir.glob("*.tif") if _clean_key(f.name) == key), None)
    lu  = next((f for f in lu_dir.glob("*.tif")  if _clean_key(f.name) == key), None)
    if not rgb or not dem or not lu:
        raise FileNotFoundError(f"Could not find TIF files for city: {city!r}")
    return rgb, dem, lu


def _extract_crop(src_rgb, src_dem, src_lu, lon, lat, crop_size=224):
    """Extract a normalised 12-channel (C, H, W) numpy array centred at (lon, lat)."""
    row, col = src_rgb.index(lon, lat)
    half     = crop_size // 2
    r_start  = int(np.clip(row - half, 0, src_rgb.height - crop_size))
    c_start  = int(np.clip(col - half, 0, src_rgb.width  - crop_size))
    win      = Window(c_start, r_start, crop_size, crop_size)

    rgb = src_rgb.read([1,2,3], window=win, boundless=True, fill_value=0).astype(np.float32) / 255.
    dem = src_dem.read(1,       window=win, boundless=True, fill_value=0).astype(np.float32)
    lu  = src_lu.read(1,        window=win, boundless=True, fill_value=0)

    dem = np.clip(dem, _DEM_MIN, _DEM_MAX)
    dem = (dem - _DEM_MIN) / (_DEM_MAX - _DEM_MIN)
    lu_oh = np.zeros((8, crop_size, crop_size), dtype=np.float32)
    for k in range(8):
        lu_oh[k] = (lu == k)

    return rgb, np.concatenate([rgb, dem[None], lu_oh], axis=0)


def _load_model(model_type: str, checkpoint: Path, device: str,
                backbone: str = "efficientnet_b3") -> nn.Module:
    from configs import SingleBranchConfig, DualBranchConfig, FiLMConfig, CrossAttnConfig, DANNConfig
    from models import build_model

    if model_type == "single":
        cfg = SingleBranchConfig(backbone=backbone)
    elif model_type == "dual":
        cfg = DualBranchConfig(backbone=backbone)
    elif model_type == "film":
        cfg = FiLMConfig(backbone=backbone)
    elif model_type == "dann":
        cfg = DANNConfig(backbone=backbone)
    elif model_type == "multitask":
        from configs import MultiTaskConfig
        cfg = MultiTaskConfig(backbone=backbone)
    else:
        cfg = CrossAttnConfig(backbone=backbone)

    model = build_model(cfg, device=device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


# ── 1. Ground-truth plot ───────────────────────────────────────────────────

def plot_ground_truth(src_rgb, samples, city: str, out_path: Path):
    print("  Rendering ground-truth overlay…")

    # Read a downsampled RGB overview
    scale = 8
    h = src_rgb.height // scale
    w = src_rgb.width  // scale
    win = Window(0, 0, src_rgb.width, src_rgb.height)
    rgb_full = src_rgb.read([1,2,3], window=win,
                             out_shape=(3, h, w),
                             resampling=rasterio.enums.Resampling.bilinear)
    # Percentile stretch handles any bit depth (uint8, uint16, float)
    rgb_arr = rgb_full.astype(np.float32)
    for c in range(3):
        p2, p98 = np.percentile(rgb_arr[c], 2), np.percentile(rgb_arr[c], 98)
        rgb_arr[c] = (rgb_arr[c] - p2) / (p98 - p2 + 1e-8)
    rgb_full = np.clip(rgb_arr.transpose(1, 2, 0), 0, 1)

    # Convert sample coordinates to pixel space (downscaled)
    xs, ys, pops = [], [], []
    for s in samples:
        sp  = s.get("start_point", {})
        lat = sp.get("lat", s.get("lat", 0.))
        lon = sp.get("lng", s.get("lon", 0.))
        try:
            row, col = src_rgb.index(lon, lat)
            xs.append(col / scale)
            ys.append(row / scale)
            pops.append(float(s["population_15min_walk"]))
        except Exception:
            pass

    pops = np.array(pops)
    norm = Normalize(vmin=np.percentile(pops, 5), vmax=np.percentile(pops, 95))
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(rgb_full)
    sc = ax.scatter(xs, ys,
                    c=pops, cmap=cmap, norm=norm,
                    s=np.clip(pops / pops.max() * 120, 10, 120),
                    alpha=0.75, edgecolors="white", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Population (15-min walk)", fraction=0.03)
    ax.set_title(f"Ground truth — {city}", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── 2. Dense prediction heatmap ────────────────────────────────────────────

def plot_prediction_heatmap(
    src_rgb, src_dem, src_lu,
    model, city: str, city_tabular: np.ndarray,
    device: str, stride: int, crop_size: int,
    out_path: Path,
):
    print(f"  Running sliding-window inference  (stride={stride})…")

    H, W     = src_rgb.height, src_rgb.width
    half     = crop_size // 2
    rows_idx = range(half, H - half, stride)
    cols_idx = range(half, W - half, stride)

    # Pre-build a grid of (row, col) centres
    centres  = [(r, c) for r in rows_idx for c in cols_idx]
    n_grid_r = len(list(rows_idx))
    n_grid_c = len(list(cols_idx))

    pred_grid = np.zeros((n_grid_r, n_grid_c), dtype=np.float32)

    tabular_t = torch.from_numpy(city_tabular).float().unsqueeze(0).to(device)
    uses_tab  = hasattr(model, "uses_tabular") or hasattr(model, "tabular_mlp")

    batch_size = 64
    crops_batch, centres_batch = [], []

    def _run_batch(crops, tab):
        x = torch.from_numpy(np.stack(crops)).float().to(device)
        with torch.no_grad():
            if uses_tab:
                t = tab.expand(len(crops), -1)
                out = model(x, t)
            else:
                out = model(x)
        # DANNDualBranch returns (pred, domain_logits) — take only the prediction
        if isinstance(out, tuple):
            out = out[0]
        log_preds = out.squeeze().cpu().numpy()
        return np.expm1(np.atleast_1d(log_preds))

    results = []
    win_cache = Window(0, 0, src_rgb.width, src_rgb.height)

    for i, (row, col) in enumerate(tqdm(centres, desc="  Sliding window")):
        r_start = int(np.clip(row - half, 0, H - crop_size))
        c_start = int(np.clip(col - half, 0, W - crop_size))
        win     = Window(c_start, r_start, crop_size, crop_size)

        try:
            rgb = src_rgb.read([1,2,3], window=win, boundless=True, fill_value=0).astype(np.float32) / 255.
            dem = src_dem.read(1,       window=win, boundless=True, fill_value=0).astype(np.float32)
            lu  = src_lu.read(1,        window=win, boundless=True, fill_value=0)
            dem = (np.clip(dem, _DEM_MIN, _DEM_MAX) - _DEM_MIN) / (_DEM_MAX - _DEM_MIN)
            lu_oh = np.zeros((8, crop_size, crop_size), dtype=np.float32)
            for k in range(8): lu_oh[k] = (lu == k)
            crop_12ch = np.concatenate([rgb, dem[None], lu_oh], axis=0)
        except Exception:
            crop_12ch = np.zeros((12, crop_size, crop_size), dtype=np.float32)

        crops_batch.append(crop_12ch)
        centres_batch.append(i)

        if len(crops_batch) == batch_size or i == len(centres) - 1:
            preds = _run_batch(crops_batch, tabular_t)
            results.extend(zip(centres_batch, preds))
            crops_batch, centres_batch = [], []

    for idx, pred in results:
        gi = idx // n_grid_c
        gj = idx %  n_grid_c
        if 0 <= gi < n_grid_r and 0 <= gj < n_grid_c:
            pred_grid[gi, gj] = pred

    # Upsample pred_grid to image resolution
    from scipy.ndimage import zoom
    zoom_r = H / n_grid_r
    zoom_c = W / n_grid_c
    pred_full = zoom(pred_grid, (zoom_r, zoom_c), order=1)
    pred_full  = gaussian_filter(pred_full, sigma=stride / 2)
    pred_full  = np.clip(pred_full, 0, TARGET_CLAMP_MAX)

    # Read RGB overview
    scale    = 8
    ho, wo   = H // scale, W // scale
    rgb_full = src_rgb.read([1,2,3],
                             out_shape=(3, ho, wo),
                             resampling=rasterio.enums.Resampling.bilinear)
    # Percentile stretch handles any bit depth (uint8, uint16, float)
    rgb_arr = rgb_full.astype(np.float32)
    for c in range(3):
        p2, p98 = np.percentile(rgb_arr[c], 2), np.percentile(rgb_arr[c], 98)
        rgb_arr[c] = (rgb_arr[c] - p2) / (p98 - p2 + 1e-8)
    rgb_full = np.clip(rgb_arr.transpose(1, 2, 0), 0, 1)
    pred_ds  = zoom(pred_full, (1/scale, 1/scale), order=1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Left: RGB only
    axes[0].imshow(rgb_full)
    axes[0].set_title("Satellite image", fontsize=13)
    axes[0].axis("off")

    # Right: RGB + heatmap overlay
    vmax = np.percentile(pred_ds[pred_ds > 0], 95) if (pred_ds > 0).any() else 1.
    norm = Normalize(vmin=0, vmax=vmax)
    axes[1].imshow(rgb_full)
    hm = axes[1].imshow(pred_ds, cmap="inferno", norm=norm, alpha=0.55)
    plt.colorbar(hm, ax=axes[1], label="Predicted population (15-min walk)", fraction=0.03)
    axes[1].set_title(f"Predicted population heatmap — {city}", fontsize=13)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── 3. Attention map (CrossAttnDualBranch only) ────────────────────────────

def plot_attention_map(
    src_rgb, src_dem, src_lu,
    model, sample: dict,
    tab_mean: np.ndarray, tab_std: np.ndarray,
    device: str, crop_size: int,
    out_path: Path,
):
    from models.dual_branch import CrossAttnDualBranch
    if not isinstance(model, CrossAttnDualBranch):
        print("  Attention map only available for CrossAttnDualBranch — skipping.")
        return

    print("  Extracting cross-attention map…")

    sp  = sample.get("start_point", {})
    lat = sp.get("lat", sample.get("lat", 0.))
    lon = sp.get("lng", sample.get("lon", 0.))
    pop = sample["population_15min_walk"]

    rgb_arr, crop_12ch = _extract_crop(src_rgb, src_dem, src_lu, lon, lat, crop_size)

    # Hook to capture attention weights: shape (B, n_heads, 1, N) → average over heads
    attn_weights = {}
    def _hook(module, inp, out):
        # out[1] = attention weight tensor (B, 1, N) averaged or per-head
        if out[1] is not None:
            attn_weights["w"] = out[1].detach().cpu()

    handle = model.cross_attn.attn.register_forward_hook(_hook)

    x = torch.from_numpy(crop_12ch).float().unsqueeze(0).to(device)
    # Tabular: raw → log1p → z-score  (must match training pipeline)
    feats  = np.array(sample.get("osm_features") or [0.] * N_TABULAR, dtype=np.float32)
    log_f  = np.log1p(feats[:N_TABULAR])
    norm_f = (log_f - tab_mean) / tab_std
    t = torch.from_numpy(norm_f).float().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(x, t)
        # DANNDualBranch returns (pred, domain_logits)
        pred_log = (result[0] if isinstance(result, tuple) else result).item()
    handle.remove()

    pred_pop = float(np.expm1(pred_log))

    # Attention weights → (1, N) or (1, n_heads, N) depending on PyTorch version
    w = attn_weights.get("w")
    if w is None:
        print("  Could not capture attention weights — skipping.")
        return

    w = w.squeeze()  # might be (N,) or (n_heads, N)
    if w.ndim == 2:
        w = w.mean(0)   # average over heads → (N,)
    w = w.numpy()

    # Determine spatial shape (should be H_feat * W_feat = 49 for 7×7)
    N_spatial = len(w)
    H_feat = int(np.sqrt(N_spatial))
    W_feat = N_spatial // H_feat
    attn_map = w[:H_feat * W_feat].reshape(H_feat, W_feat)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to crop_size × crop_size
    from scipy.ndimage import zoom as nd_zoom
    attn_up = nd_zoom(attn_map, (crop_size / H_feat, crop_size / W_feat), order=1)
    attn_up = gaussian_filter(attn_up, sigma=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rgb_disp = np.clip(rgb_arr.transpose(1,2,0), 0, 1)

    # Panel 1: RGB crop
    axes[0].imshow(rgb_disp)
    axes[0].set_title("RGB crop  (224×224)", fontsize=12)
    axes[0].axis("off")

    # Panel 2: attention map only
    im = axes[1].imshow(attn_map, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Cross-attention weights  ({H_feat}×{W_feat})", fontsize=12)
    axes[1].set_xlabel("Column patch index")
    axes[1].set_ylabel("Row patch index")

    # Panel 3: overlay
    axes[2].imshow(rgb_disp)
    ov = axes[2].imshow(attn_up, cmap="hot", alpha=0.55)
    plt.colorbar(ov, ax=axes[2], fraction=0.046, label="Attention weight")
    axes[2].set_title(
        f"Attention overlay\n"
        f"GT: {pop:,.0f} pop  |  Pred: {pred_pop:,.0f} pop",
        fontsize=12,
    )
    axes[2].axis("off")

    fig.suptitle(f"{sample['city']}  —  ({lat:.4f}, {lon:.4f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",       required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model",      default="crossattn",
                        choices=["single", "dual", "film", "crossattn", "dann", "multitask"])
    parser.add_argument("--backbone",   default="efficientnet_b3",
                        help="Backbone name (default: efficientnet_b3)")
    parser.add_argument("--json",       required=True)
    parser.add_argument("--base",       required=True)
    parser.add_argument("--out",        default="visualizations")
    parser.add_argument("--stride",     type=int, default=56,
                        help="Sliding window stride in pixels (lower = denser)")
    parser.add_argument("--crop",       type=int, default=224)
    parser.add_argument("--cache",      default=None,
                        help="Path to the preprocessed cache directory "
                             "(used to load tab_mean/tab_std from stats.json). "
                             "Required for correct tabular feature normalisation.")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the (slow) dense prediction heatmap")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir  = Path(args.out);  out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base)
    city_slug = args.city.replace(", ", "_").replace(" ", "_")

    # Load tabular normalisation stats from the preprocessed cache
    tab_mean: np.ndarray = np.zeros(N_TABULAR, dtype=np.float32)
    tab_std:  np.ndarray = np.ones(N_TABULAR,  dtype=np.float32)
    if args.cache is not None:
        stats_path = Path(args.cache) / "stats.json"
        if stats_path.exists():
            with open(stats_path) as _f:
                _stats = json.load(_f)
            tab_mean = np.array(_stats["tab_mean"], dtype=np.float32)
            tab_std  = np.array(_stats["tab_std"],  dtype=np.float32)
            print(f"Loaded tabular stats from {stats_path}")
        else:
            print(f"WARNING: --cache provided but stats.json not found at {stats_path}")
            print("         Tabular features will NOT be z-scored — predictions may be off.")
    else:
        print("WARNING: --cache not provided. Tabular features will NOT be z-scored.")
        print("         Pass --cache /workspace/PopulationDataset/cache for correct results.")

    print(f"\n{'='*60}")
    print(f"  City       : {args.city}")
    print(f"  Model      : {args.model}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # Load JSON
    with open(args.json, "rb") as f:
        all_data = json.load(f)
    samples = [s for s in all_data
               if s.get("city") == args.city
               and s.get("population_15min_walk") is not None]
    print(f"Found {len(samples)} samples for {args.city}")
    if not samples:
        print("No samples found — check city name (exact match required).")
        return

    # Open TIF files
    rgb_path, dem_path, lu_path = _find_city_files(base_dir, args.city)
    src_rgb = rasterio.open(rgb_path)
    src_dem = rasterio.open(dem_path)
    src_lu  = rasterio.open(lu_path)
    print(f"TIF size: {src_rgb.width} × {src_rgb.height} px")

    # Load model
    model = _load_model(args.model, Path(args.checkpoint), device, backbone=args.backbone)

    # City-average tabular features for sliding window
    # Pipeline must match training: raw → log1p → z-score with training-split stats
    tab_arr = np.array(
        [s.get("osm_features") or [0.]*N_TABULAR for s in samples],
        dtype=np.float32,
    )
    log_tab = np.log1p(tab_arr)
    city_tabular = ((log_tab.mean(axis=0) - tab_mean) / tab_std).astype(np.float32)

    # ── 1. Ground truth ───────────────────────────────────────────────────
    plot_ground_truth(
        src_rgb, samples, args.city,
        out_dir / f"{city_slug}_groundtruth.png",
    )

    # ── 2. Dense prediction heatmap ───────────────────────────────────────
    if not args.no_heatmap:
        plot_prediction_heatmap(
            src_rgb, src_dem, src_lu,
            model, args.city, city_tabular,
            device, args.stride, args.crop,
            out_dir / f"{city_slug}_heatmap.png",
        )

    # ── 3. Attention map (pick sample with median population) ─────────────
    pops   = [s["population_15min_walk"] for s in samples]
    median_idx = int(np.argsort(pops)[len(pops) // 2])
    plot_attention_map(
        src_rgb, src_dem, src_lu,
        model, samples[median_idx],
        tab_mean, tab_std,
        device, args.crop,
        out_dir / f"{city_slug}_attention.png",
    )

    src_rgb.close(); src_dem.close(); src_lu.close()
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
