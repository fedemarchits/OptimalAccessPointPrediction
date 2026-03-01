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
import geopandas as gpd
from shapely.geometry import Point

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
_BUFFER_RADIUS_M     = 1120
_BUFFER_AREA_M2      = np.pi * _BUFFER_RADIUS_M ** 2


# ── OSM feature extraction (mirrors extract_osm_tabular_features.py) ──────────

def _load_city_osm(city_name: str, osm_dir: Path) -> dict:
    """Load all OSM GeoPackage layers for a city, reprojected to EPSG:3857."""
    city_dir = osm_dir / city_name.replace(", ", "_").replace(" ", "_")
    layers = {}
    for name in ("road_nodes", "road_edges", "buildings", "pois", "transport"):
        path = city_dir / f"{name}.gpkg"
        if not path.exists():
            layers[name] = gpd.GeoDataFrame()
            continue
        try:
            gdf = gpd.read_file(path)
            layers[name] = (gdf.to_crs("EPSG:3857")
                            if not gdf.empty and gdf.crs is not None
                            else gpd.GeoDataFrame())
        except Exception:
            layers[name] = gpd.GeoDataFrame()
    return layers


def _clip_to_buffer(gdf: gpd.GeoDataFrame, buf) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    try:
        idx = list(gdf.sindex.intersection(buf.bounds))
        if not idx:
            return gdf.iloc[[]]
        candidates = gdf.iloc[idx]
        return candidates[candidates.geometry.intersects(buf)]
    except Exception:
        return gdf.iloc[[]]


def _extract_osm_features(lon: float, lat: float, city_osm: dict) -> np.ndarray:
    """Compute all 15 tabular OSM features for (lon, lat) within 1120 m buffer."""
    pt  = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857")
    buf = pt.buffer(_BUFFER_RADIUS_M).iloc[0]
    feats = np.zeros(N_TABULAR, dtype=np.float32)

    edges = _clip_to_buffer(city_osm.get("road_edges", gpd.GeoDataFrame()), buf)
    if not edges.empty:
        lengths  = edges["length"] if "length" in edges.columns else edges.geometry.length
        feats[0] = float(lengths.sum())
        feats[1] = float(len(edges))
        feats[2] = float(feats[0] / _BUFFER_AREA_M2)

    nodes = _clip_to_buffer(city_osm.get("road_nodes", gpd.GeoDataFrame()), buf)
    if not nodes.empty:
        feats[3] = float(len(nodes))
        feats[4] = (float((nodes["street_count"] >= 3).sum())
                    if "street_count" in nodes.columns
                    else float(len(nodes) * 0.5))

    bldgs = _clip_to_buffer(city_osm.get("buildings", gpd.GeoDataFrame()), buf)
    if not bldgs.empty:
        feats[5] = float(len(bldgs))
        poly = bldgs.geometry.type.isin(["Polygon", "MultiPolygon"])
        if poly.any():
            area     = float(bldgs.loc[poly, "geometry"].area.sum())
            feats[6] = area
            feats[7] = area / _BUFFER_AREA_M2

    pois = _clip_to_buffer(city_osm.get("pois", gpd.GeoDataFrame()), buf)
    if not pois.empty:
        feats[8] = float(len(pois))
        for j, tag in enumerate(["amenity", "shop", "leisure", "tourism", "office"], 9):
            if tag in pois.columns:
                feats[j] = float(pois[tag].notna().sum())

    transport = _clip_to_buffer(city_osm.get("transport", gpd.GeoDataFrame()), buf)
    feats[14] = float(len(transport))

    return feats


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
    model, city: str,
    city_log_mean: float, city_log_std: float,
    tab_mean: np.ndarray, tab_std: np.ndarray,
    device: str, stride: int, crop_size: int,
    out_path: Path,
    osm_dir: Path = None,
    city_tabular_fallback: np.ndarray = None,
):
    print(f"  Running sliding-window inference  (stride={stride})…")

    uses_tab = hasattr(model, "uses_tabular") or hasattr(model, "tabular_mlp")

    # Load OSM layers once for the city (used to compute per-point tabular features)
    city_osm = None
    if uses_tab and osm_dir is not None:
        osm_city_dir = osm_dir / city.replace(", ", "_").replace(" ", "_")
        if osm_city_dir.exists():
            print(f"  Loading OSM layers for {city}…")
            city_osm = _load_city_osm(city, osm_dir)
            print(f"  OSM layers loaded — will compute per-point tabular features.")
        else:
            print(f"  WARNING: OSM dir not found for {city}, falling back to city-average tabular.")

    H, W     = src_rgb.height, src_rgb.width
    half     = crop_size // 2
    rows_idx = range(half, H - half, stride)
    cols_idx = range(half, W - half, stride)

    centres  = [(r, c) for r in rows_idx for c in cols_idx]
    n_grid_r = len(list(rows_idx))
    n_grid_c = len(list(cols_idx))

    pred_grid = np.zeros((n_grid_r, n_grid_c), dtype=np.float32)
    batch_size = 64
    crops_batch, tabs_batch, centres_batch = [], [], []

    def _run_batch(crops, tabs):
        x = torch.from_numpy(np.stack(crops)).float().to(device)
        with torch.no_grad():
            if uses_tab:
                t = torch.from_numpy(np.stack(tabs)).float().to(device)
                out = model(x, t)
            else:
                out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        z_preds = out.squeeze().cpu().numpy()
        return np.expm1(np.atleast_1d(z_preds) * city_log_std + city_log_mean)

    results = []

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

        # Per-point tabular features
        if uses_tab:
            if city_osm is not None:
                try:
                    lon, lat = src_rgb.xy(row, col)
                    raw_feats = _extract_osm_features(lon, lat, city_osm)
                    tab = (np.log1p(raw_feats) - tab_mean) / tab_std
                except Exception:
                    tab = city_tabular_fallback
            else:
                tab = city_tabular_fallback
        else:
            tab = np.zeros(N_TABULAR, dtype=np.float32)

        crops_batch.append(crop_12ch)
        tabs_batch.append(tab)
        centres_batch.append(i)

        if len(crops_batch) == batch_size or i == len(centres) - 1:
            preds = _run_batch(crops_batch, tabs_batch)
            results.extend(zip(centres_batch, preds))
            crops_batch, tabs_batch, centres_batch = [], [], []

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

def _attention_for_sample(model, src_rgb, src_dem, src_lu,
                           sample, tab_mean, tab_std,
                           city_log_mean, city_log_std,
                           device, crop_size):
    """Extract cross-attention map for one sample. Returns (rgb_disp, attn_map, attn_up, gt_pop, pred_pop, lat, lon) or None."""
    sp  = sample.get("start_point", {})
    lat = sp.get("lat", sample.get("lat", 0.))
    lon = sp.get("lng", sample.get("lon", 0.))
    pop = sample["population_15min_walk"]

    try:
        rgb_arr, crop_12ch = _extract_crop(src_rgb, src_dem, src_lu, lon, lat, crop_size)
    except Exception:
        return None

    attn_weights = {}
    def _hook(module, inp, out):
        if out[1] is not None:
            attn_weights["w"] = out[1].detach().cpu()

    handle = model.cross_attn.attn.register_forward_hook(_hook)
    x = torch.from_numpy(crop_12ch).float().unsqueeze(0).to(device)
    feats  = np.array(sample.get("osm_features") or [0.] * N_TABULAR, dtype=np.float32)
    norm_f = (np.log1p(feats[:N_TABULAR]) - tab_mean) / tab_std
    t = torch.from_numpy(norm_f).float().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(x, t)
        z_pred = (result[0] if isinstance(result, tuple) else result).item()
    handle.remove()

    pred_pop = float(np.expm1(z_pred * city_log_std + city_log_mean))

    w = attn_weights.get("w")
    if w is None:
        return None
    w = w.squeeze()
    if w.ndim == 2:
        w = w.mean(0)
    w = w.numpy()

    N_spatial = len(w)
    H_feat = int(np.sqrt(N_spatial))
    W_feat = N_spatial // H_feat
    attn_map = w[:H_feat * W_feat].reshape(H_feat, W_feat)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    from scipy.ndimage import zoom as nd_zoom
    attn_up = nd_zoom(attn_map, (crop_size / H_feat, crop_size / W_feat), order=1)
    attn_up = gaussian_filter(attn_up, sigma=2)

    rgb_disp = np.clip(rgb_arr.transpose(1, 2, 0), 0, 1)
    return rgb_disp, attn_map, attn_up, pop, pred_pop, lat, lon, H_feat, W_feat


def plot_attention_map(
    src_rgb, src_dem, src_lu,
    model, samples: list,
    tab_mean: np.ndarray, tab_std: np.ndarray,
    city_log_mean: float, city_log_std: float,
    device: str, crop_size: int,
    out_path: Path,
    n_patches: int = 5,
):
    """
    Render cross-attention maps for n_patches samples spread across the
    population distribution (10/30/50/70/90 percentile).
    Each row: RGB crop | attention grid | overlay.
    """
    from models.dual_branch import CrossAttnDualBranch
    if not isinstance(model, CrossAttnDualBranch):
        print("  Attention map only available for CrossAttnDualBranch — skipping.")
        return

    print(f"  Extracting cross-attention maps for {n_patches} patches…")

    # Pick samples at evenly-spaced population percentiles
    pops    = np.array([s["population_15min_walk"] for s in samples])
    order   = np.argsort(pops)
    picks   = np.linspace(0, len(order) - 1, n_patches, dtype=int)
    chosen  = [samples[order[p]] for p in picks]

    rows = []
    for s in chosen:
        res = _attention_for_sample(model, src_rgb, src_dem, src_lu,
                                    s, tab_mean, tab_std,
                                    city_log_mean, city_log_std,
                                    device, crop_size)
        if res is not None:
            rows.append(res)

    if not rows:
        print("  Could not capture attention weights for any sample — skipping.")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Cross-attention maps — {chosen[0]['city']}", fontsize=14, fontweight="bold")

    for i, (rgb_disp, attn_map, attn_up, gt_pop, pred_pop, lat, lon, H_feat, W_feat) in enumerate(rows):
        axes[i, 0].imshow(rgb_disp)
        axes[i, 0].set_title(f"RGB crop  ({lat:.4f}, {lon:.4f})\nGT: {gt_pop:,.0f}", fontsize=10)
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(attn_map, cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
        axes[i, 1].set_title(f"Cross-attention  ({H_feat}×{W_feat})", fontsize=10)
        axes[i, 1].set_xlabel("Column patch")
        axes[i, 1].set_ylabel("Row patch")

        axes[i, 2].imshow(rgb_disp)
        ov = axes[i, 2].imshow(attn_up, cmap="hot", alpha=0.55)
        plt.colorbar(ov, ax=axes[i, 2], fraction=0.046, label="Attention")
        axes[i, 2].set_title(f"Overlay  |  Pred: {pred_pop:,.0f}", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
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
    parser.add_argument("--osm-dir",    default=None,
                        help="Path to OSM features directory (e.g. data/osm_features). "
                             "If provided, per-point OSM features are computed for every "
                             "sliding-window position instead of using city average.")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the (slow) dense prediction heatmap")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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

    # Per-city target z-score stats for correct denormalisation
    city_log_mean = 0.0
    city_log_std  = 1.0
    if args.cache is not None:
        stats_path = Path(args.cache) / "stats.json"
        if stats_path.exists():
            with open(stats_path) as _f:
                _stats = json.load(_f)
            city_stats = _stats.get("city_log_stats", {}).get(args.city, {})
            city_log_mean = float(city_stats.get("mean", 0.0))
            city_log_std  = float(city_stats.get("std",  1.0))
            print(f"City z-score stats: mean={city_log_mean:.3f}, std={city_log_std:.3f}")

    # ── 1. Ground truth ───────────────────────────────────────────────────
    plot_ground_truth(
        src_rgb, samples, args.city,
        out_dir / f"{city_slug}_groundtruth.png",
    )

    # ── 2. Dense prediction heatmap ───────────────────────────────────────
    if not args.no_heatmap:
        plot_prediction_heatmap(
            src_rgb, src_dem, src_lu,
            model, args.city,
            city_log_mean, city_log_std,
            tab_mean, tab_std,
            device, args.stride, args.crop,
            out_dir / f"{city_slug}_heatmap.png",
            osm_dir=Path(args.osm_dir) if args.osm_dir else None,
            city_tabular_fallback=city_tabular,
        )

    # ── 3. Attention map (5 patches across population distribution) ───────
    plot_attention_map(
        src_rgb, src_dem, src_lu,
        model, samples,
        tab_mean, tab_std,
        city_log_mean, city_log_std,
        device, args.crop,
        out_dir / f"{city_slug}_attention.png",
        n_patches=5,
    )

    src_rgb.close(); src_dem.close(); src_lu.close()
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
