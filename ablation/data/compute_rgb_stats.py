"""
Compute per-city RGB statistics and add them to an existing cache.

Run this once on an already-preprocessed cache to enable per-city RGB
normalisation in CachedDataset without re-running the full preprocessing.

What it does
────────────
For each city, the RGB raster is read at ~1/8 resolution (fast — a few
seconds per city).  Per-channel mean and std in [0, 1] space are computed
from all valid (non-black) pixels, then written out as two arrays:

  cache/rgb_city_mean.npy   (N, 3)  float32  — per-sample R/G/B means
  cache/rgb_city_std.npy    (N, 3)  float32  — per-sample R/G/B stds
  cache/stats.json          updated with "city_rgb_stats" dict

CachedDataset picks these up automatically on next load.

Usage:
    python ablation/data/compute_rgb_stats.py \\
        --cache /workspace/PopulationDataset/cache \\
        --base  /workspace/PopulationDataset
"""

from __future__ import annotations

import argparse
import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.enums
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ImageNet fallback used when a raster cannot be opened
_IMAGENET = {
    "r_mean": 0.485, "g_mean": 0.456, "b_mean": 0.406,
    "r_std":  0.229, "g_std":  0.224, "b_std":  0.225,
}


def _clean_key(s: str) -> str:
    s = str(s).replace(".tif", "").replace(".tiff", "")
    for suf in ["_LandUse", "LandUse", "_DEM", "DEM", "_10m", "10m"]:
        s = s.replace(suf, "")
    s = s.lower()
    for ch, rep in {"ø": "o", "å": "a", "æ": "ae", "ö": "o",
                    "ü": "u", "ä": "a", "é": "e", "è": "e"}.items():
        s = s.replace(ch, rep)
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return "".join(c for c in s if c.isalnum())


def _city_rgb_stats(src_rgb) -> dict:
    """
    Compute per-channel mean/std from a downsampled raster (fast).
    Returns {r_mean, g_mean, b_mean, r_std, g_std, b_std} in [0, 1].
    """
    H, W = src_rgb.height, src_rgb.width
    # Target ~512 px on the shorter side — typically 4× downsampling
    scale = max(1, min(H, W) // 512)
    out_h, out_w = max(1, H // scale), max(1, W // scale)
    try:
        data = src_rgb.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32) / 255.0   # (3, H', W')
    except Exception:
        return dict(_IMAGENET)

    stats = {}
    for ch, name in enumerate(["r", "g", "b"]):
        pixels = data[ch].ravel()
        valid  = pixels[pixels > 0.01]   # exclude no-data (black) pixels
        if len(valid) < 100:
            valid = pixels
        stats[f"{name}_mean"] = float(valid.mean())
        stats[f"{name}_std"]  = float(max(float(valid.std()), 1e-4))
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add per-city RGB stats to an existing preprocessing cache."
    )
    parser.add_argument("--cache", required=True, help="Path to cache dir (contains index.json)")
    parser.add_argument("--base",  required=True, help="Base dataset dir (contains satellite_images/)")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    rgb_dir    = Path(args.base) / "satellite_images"

    # ── Load existing cache metadata ─────────────────────────────────────────
    with open(cache_path / "index.json") as f:
        index = json.load(f)
    N = len(index)
    print(f"Cache has {N} samples.")

    with open(cache_path / "stats.json") as f:
        stats_json = json.load(f)

    # ── Build city → RGB raster mapping ──────────────────────────────────────
    rgb_map = {_clean_key(f.name): f for f in rgb_dir.glob("*.tif")}
    cities  = sorted({e["city"] for e in index})
    print(f"Computing RGB stats for {len(cities)} cities "
          f"({len(rgb_map)} rasters found)…")

    # ── Per-city stats ────────────────────────────────────────────────────────
    city_rgb_stats: dict[str, dict] = {}
    n_fallback = 0

    for city in tqdm(cities, unit="city"):
        key = _clean_key(city)
        if key not in rgb_map:
            tqdm.write(f"  ⚠ No raster for '{city}' — ImageNet fallback")
            city_rgb_stats[city] = dict(_IMAGENET)
            n_fallback += 1
            continue
        try:
            with rasterio.open(rgb_map[key]) as src:
                city_rgb_stats[city] = _city_rgb_stats(src)
        except Exception as e:
            tqdm.write(f"  ⚠ Error for '{city}': {e} — ImageNet fallback")
            city_rgb_stats[city] = dict(_IMAGENET)
            n_fallback += 1

    if n_fallback:
        print(f"  {n_fallback}/{len(cities)} cities used ImageNet fallback.")

    # ── Build per-sample arrays ───────────────────────────────────────────────
    rgb_mean_arr = np.zeros((N, 3), dtype=np.float32)
    rgb_std_arr  = np.zeros((N, 3), dtype=np.float32)

    for i, entry in enumerate(index):
        s = city_rgb_stats[entry["city"]]
        rgb_mean_arr[i] = [s["r_mean"], s["g_mean"], s["b_mean"]]
        rgb_std_arr[i]  = [s["r_std"],  s["g_std"],  s["b_std"]]

    np.save(cache_path / "rgb_city_mean.npy", rgb_mean_arr)
    np.save(cache_path / "rgb_city_std.npy",  rgb_std_arr)

    # ── Update stats.json ─────────────────────────────────────────────────────
    stats_json["city_rgb_stats"] = city_rgb_stats
    with open(cache_path / "stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nWrote:")
    print(f"  {cache_path}/rgb_city_mean.npy  {rgb_mean_arr.nbytes / 1e6:.1f} MB")
    print(f"  {cache_path}/rgb_city_std.npy   {rgb_std_arr.nbytes  / 1e6:.1f} MB")
    print(f"  {cache_path}/stats.json  (updated)")
    print(f"\nExample city stats:")
    for city in list(city_rgb_stats)[:3]:
        s = city_rgb_stats[city]
        print(f"  {city[:40]:<40}  "
              f"R {s['r_mean']:.3f}±{s['r_std']:.3f}  "
              f"G {s['g_mean']:.3f}±{s['g_std']:.3f}  "
              f"B {s['b_mean']:.3f}±{s['b_std']:.3f}")


if __name__ == "__main__":
    main()
