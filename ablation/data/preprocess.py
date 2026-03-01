"""
One-time dataset preprocessing.

Reads all samples from the GeoTIFF files and saves them as numpy
memory-mapped arrays so that training can load crops at RAM/SSD speed
instead of re-reading GeoTIFFs on every epoch.

Jitter augmentation is preserved: crops are extracted at
  crop_size + 2 * jitter_pixels  (default 304×304)
so that the CachedDataset can apply a random 224×224 sub-crop each time.

Output layout in <cache_dir>/
  rgb.dat          (N, 3, PAD, PAD)  uint8    raw RGB
  dem.dat          (N, 1, PAD, PAD)  float32  DEM normalised to [0,1]
  landuse.dat      (N, 1, PAD, PAD)  uint8    class label 0-7
  targets.npy      (N,)              float32  log1p population (or raw)
  tabular.npy      (N, 15)           float32  log1p + z-score OSM features
  stats.json                         normalisation constants
  index.json                         [{city, cluster, split}, ...]

Usage:
    python ablation/data/preprocess.py \\
        --json  /workspace/PopulationDataset/final_clustered_samples.json \\
        --base  /workspace/PopulationDataset \\
        --out   /workspace/PopulationDataset/cache
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants matching dataset.py ─────────────────────────────────────────────
_DEM_MIN = -50.0
_DEM_MAX = 3_000.0
N_TABULAR = 15
N_CLASSES  = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_key(s: str) -> str:
    import unicodedata
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


def _build_city_mapping(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    rgb_dir = base_dir / "satellite_images"
    dem_dir = base_dir / "dem_height"
    lu_dir  = base_dir / "segmentation_land_use"
    dem_map = {_clean_key(f.name): f for f in dem_dir.glob("*.tif")}
    lu_map  = {_clean_key(f.name): f for f in lu_dir.glob("*.tif")}
    mapping = {}
    for rgb_path in rgb_dir.glob("*.tif"):
        key = _clean_key(rgb_path.name)
        if key in dem_map and key in lu_map:
            mapping[key] = {"rgb": rgb_path, "dem": dem_map[key], "landuse": lu_map[key]}
    return mapping


_SPLIT_FILE = Path(__file__).parent / "city_split.json"


def _city_split(data: list, train_ratio: float, val_ratio: float,
                seed: int) -> Tuple[set, set, set]:
    # Use the canonical fixed split if it has been generated
    if _SPLIT_FILE.exists():
        with open(_SPLIT_FILE) as f:
            sp = json.load(f)
        print(f"  Loaded fixed city split from {_SPLIT_FILE}")
        return set(sp["train"]), set(sp["val"]), set(sp["test"])

    # Fallback: compute deterministically (run generate_split.py to fix it)
    print("  WARNING: city_split.json not found — computing split dynamically.")
    print("  Run: python data/generate_split.py --json <path> to fix it.")
    cities = sorted({s["city"] for s in data})
    rng = np.random.default_rng(seed)
    rng.shuffle(cities)
    n    = len(cities)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return set(cities[:n_tr]), set(cities[n_tr:n_tr + n_va]), set(cities[n_tr + n_va:])


def _city_rgb_stats(src_rgb) -> dict:
    """
    Compute per-channel RGB mean/std from a downsampled raster in [0, 1].
    Returns {r_mean, g_mean, b_mean, r_std, g_std, b_std}.
    """
    H, W = src_rgb.height, src_rgb.width
    scale = max(1, min(H, W) // 512)
    out_h, out_w = max(1, H // scale), max(1, W // scale)
    try:
        import rasterio.enums
        data = src_rgb.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32) / 255.0
    except Exception:
        return {"r_mean": 0.485, "g_mean": 0.456, "b_mean": 0.406,
                "r_std":  0.229, "g_std":  0.224, "b_std":  0.225}
    stats = {}
    for ch, name in enumerate(["r", "g", "b"]):
        pixels = data[ch].ravel()
        valid  = pixels[pixels > 0.01]
        if len(valid) < 100:
            valid = pixels
        stats[f"{name}_mean"] = float(valid.mean())
        stats[f"{name}_std"]  = float(max(float(valid.std()), 1e-4))
    return stats


def _read_crop(src_rgb, src_dem, src_lu,
               lon: float, lat: float,
               pad_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract a pad_size × pad_size crop centred at (lon, lat)."""
    try:
        row, col = src_rgb.index(lon, lat)
    except Exception:
        return None

    half    = pad_size // 2
    r_start = int(np.clip(row - half, 0, src_rgb.height - pad_size))
    c_start = int(np.clip(col - half, 0, src_rgb.width  - pad_size))
    window  = Window(c_start, r_start, pad_size, pad_size)

    try:
        rgb = src_rgb.read([1, 2, 3], window=window, boundless=True, fill_value=0)  # (3,H,W) uint8
        dem = src_dem.read(1,         window=window, boundless=True, fill_value=0)  # (H,W)
        lu  = src_lu.read(1,          window=window, boundless=True, fill_value=0)  # (H,W)
    except Exception:
        return None

    # Normalise DEM to [0,1]
    dem = np.clip(dem.astype(np.float32), _DEM_MIN, _DEM_MAX)
    dem = (dem - _DEM_MIN) / (_DEM_MAX - _DEM_MIN)

    return rgb.astype(np.uint8), dem[np.newaxis].astype(np.float32), lu[np.newaxis].astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess(
    json_file: str,
    base_dir: str,
    out_dir: str,
    crop_size: int = 224,
    jitter_pixels: int = 40,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    target_clamp_max: float = 80_000.0,
):
    json_path = Path(json_file)
    base_path = Path(base_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pad_size = crop_size + 2 * jitter_pixels   # e.g. 304

    print("=" * 65)
    print("Dataset Preprocessor")
    print(f"  JSON        : {json_path}")
    print(f"  Base dir    : {base_path}")
    print(f"  Output      : {out_path}")
    print(f"  Crop size   : {crop_size} px  (padded {pad_size} px with ±{jitter_pixels} jitter)")
    print("=" * 65)

    # ── Load JSON ─────────────────────────────────────────────────────────────
    with open(json_path, "rb") as f:
        raw_data = json.load(f)
    raw_data = [s for s in raw_data if s.get("population_15min_walk") is not None]
    print(f"\nLoaded {len(raw_data)} samples with valid targets")

    # ── City mapping ──────────────────────────────────────────────────────────
    city_map = _build_city_mapping(base_path)
    print(f"Mapped {len(city_map)} complete cities (RGB + DEM + LandUse)")

    # Filter to mappable cities only
    data = [s for s in raw_data if _clean_key(s["city"]) in city_map]
    print(f"Samples in mappable cities: {len(data)}")

    # ── City split (same seed as training) ────────────────────────────────────
    train_cities, val_cities, test_cities = _city_split(
        data, train_ratio, val_ratio, random_seed
    )
    split_map = {}
    for s in data:
        c = s["city"]
        split_map[c] = ("train" if c in train_cities
                        else "val" if c in val_cities else "test")

    # ── Compute normalisation stats ────────────────────────────────────────────
    train_samples = [s for s in data if split_map[s["city"]] == "train"]

    # Per-city log1p stats — computed from ALL samples in each city (no leakage
    # since cities are fully disjoint across splits)
    from collections import defaultdict
    city_raw_pops: Dict[str, list] = defaultdict(list)
    for s in data:
        city_raw_pops[s["city"]].append(
            float(np.clip(s["population_15min_walk"], 0.0, target_clamp_max))
        )
    city_log_stats: Dict[str, Dict[str, float]] = {}
    for city, pops in city_raw_pops.items():
        log_pops = np.log1p(np.array(pops, dtype=np.float32))
        city_log_stats[city] = {
            "mean": float(log_pops.mean()),
            # Clamp std to avoid division-by-zero for single-sample cities
            "std":  float(max(float(log_pops.std()), 1e-4)),
        }
    print(f"\nPer-city z-score stats computed for {len(city_log_stats)} cities")

    # Global log1p stats (kept for backwards-compat in stats.json)
    raw_targets = np.array([s["population_15min_walk"] for s in train_samples], dtype=np.float32)
    raw_targets = np.clip(raw_targets, 0.0, target_clamp_max)
    log_targets = np.log1p(raw_targets)
    log_mean = float(log_targets.mean())
    log_std  = float(log_targets.std())
    print(f"Global target stats (train, log1p):  mean={log_mean:.3f}  std={log_std:.3f}")

    # Tabular stats (log1p → z-score, from train split)
    raw_tab = np.array(
        [s.get("osm_features") or [0.0] * N_TABULAR for s in train_samples],
        dtype=np.float32,
    )
    log_tab = np.log1p(raw_tab)
    tab_mean = log_tab.mean(axis=0)
    tab_std  = log_tab.std(axis=0) + 1e-8
    print(f"Tabular stats (train):  mean avg={tab_mean.mean():.3f}")

    # ── Allocate memmaps ──────────────────────────────────────────────────────
    N = len(data)
    print(f"\nAllocating memmaps for {N} samples  ({pad_size}×{pad_size} crops)…")

    rgb_mm = np.lib.format.open_memmap(
        out_path / "rgb.dat", mode="w+", dtype=np.uint8,
        shape=(N, 3, pad_size, pad_size))
    dem_mm = np.lib.format.open_memmap(
        out_path / "dem.dat", mode="w+", dtype=np.float32,
        shape=(N, 1, pad_size, pad_size))
    lu_mm  = np.lib.format.open_memmap(
        out_path / "landuse.dat", mode="w+", dtype=np.uint8,
        shape=(N, 1, pad_size, pad_size))

    targets_arr       = np.zeros(N, dtype=np.float32)
    city_log_mean_arr = np.zeros(N, dtype=np.float32)
    city_log_std_arr  = np.zeros(N, dtype=np.float32)
    tabular_arr       = np.zeros((N, N_TABULAR), dtype=np.float32)
    rgb_city_mean_arr = np.zeros((N, 3), dtype=np.float32)
    rgb_city_std_arr  = np.zeros((N, 3), dtype=np.float32)
    city_rgb_stats_all: Dict[str, dict] = {}
    index_list        = []

    # ── Extract crops city by city ────────────────────────────────────────────
    cities_ordered = sorted({s["city"] for s in data})
    sample_by_city: Dict[str, List[Tuple[int, dict]]] = {}
    for i, s in enumerate(data):
        sample_by_city.setdefault(s["city"], []).append((i, s))

    n_failed = 0

    for city in tqdm(cities_ordered, desc="Cities", unit="city"):
        key = _clean_key(city)
        files = city_map[key]
        try:
            src_rgb = rasterio.open(files["rgb"])
            src_dem = rasterio.open(files["dem"])
            src_lu  = rasterio.open(files["landuse"])
        except Exception as e:
            tqdm.write(f"  ⚠ Cannot open {city}: {e}")
            n_failed += len(sample_by_city[city])
            continue

        rgb_stats = _city_rgb_stats(src_rgb)
        city_rgb_stats_all[city] = rgb_stats
        _city_rgb_m = np.array([rgb_stats["r_mean"], rgb_stats["g_mean"], rgb_stats["b_mean"]], dtype=np.float32)
        _city_rgb_s = np.array([rgb_stats["r_std"],  rgb_stats["g_std"],  rgb_stats["b_std"]],  dtype=np.float32)

        for i, s in sample_by_city[city]:
            rgb_city_mean_arr[i] = _city_rgb_m
            rgb_city_std_arr[i]  = _city_rgb_s
            sp  = s.get("start_point", {})
            lat = sp.get("lat", s.get("lat", 0.0))
            lon = sp.get("lng", s.get("lon", 0.0))

            result = _read_crop(src_rgb, src_dem, src_lu, lon, lat, pad_size)
            if result is None:
                n_failed += 1
                # Store zeros — sample will be filtered at training time via index
                index_list.append({"city": city, "cluster": s.get("cluster", 0),
                                   "split": split_map[city], "valid": False})
            else:
                rgb_mm[i], dem_mm[i], lu_mm[i] = result
                index_list.append({"city": city, "cluster": s.get("cluster", 0),
                                   "split": split_map[city], "valid": True})

            # Target — per-city z-score of log1p(population)
            raw_t  = float(np.clip(s["population_15min_walk"], 0.0, target_clamp_max))
            log_t  = float(np.log1p(raw_t))
            c_mean = city_log_stats[city]["mean"]
            c_std  = city_log_stats[city]["std"]
            targets_arr[i]       = (log_t - c_mean) / c_std
            city_log_mean_arr[i] = c_mean
            city_log_std_arr[i]  = c_std

            # Tabular
            feats = s.get("osm_features") or [0.0] * N_TABULAR
            log_f = np.log1p(np.array(feats, dtype=np.float32))
            tabular_arr[i] = (log_f - tab_mean) / tab_std

        src_rgb.close(); src_dem.close(); src_lu.close()

    # ── Flush and save ────────────────────────────────────────────────────────
    del rgb_mm, dem_mm, lu_mm   # flush memmaps

    np.save(out_path / "targets.npy",        targets_arr)
    np.save(out_path / "city_log_mean.npy",  city_log_mean_arr)
    np.save(out_path / "city_log_std.npy",   city_log_std_arr)
    np.save(out_path / "tabular.npy",        tabular_arr)
    np.save(out_path / "rgb_city_mean.npy",  rgb_city_mean_arr)
    np.save(out_path / "rgb_city_std.npy",   rgb_city_std_arr)

    with open(out_path / "index.json", "w") as f:
        json.dump(index_list, f)

    with open(out_path / "stats.json", "w") as f:
        json.dump({
            "log_mean": log_mean, "log_std": log_std,
            "tab_mean": tab_mean.tolist(), "tab_std": tab_std.tolist(),
            "crop_size": crop_size, "pad_size": pad_size,
            "jitter_pixels": jitter_pixels,
            "target_clamp_max": target_clamp_max,
            "n_samples": N, "n_failed": n_failed,
            "target_norm": "city_zscore",
            "city_log_stats": city_log_stats,
            "city_rgb_stats": city_rgb_stats_all,
        }, f, indent=2)

    # Summary
    splits = {"train": 0, "val": 0, "test": 0}
    valid  = sum(1 for e in index_list if e["valid"])
    for e in index_list:
        splits[e["split"]] += 1

    print(f"\n{'='*65}")
    print(f"Done!  {valid}/{N} samples extracted successfully  ({n_failed} failed)")
    print(f"  train={splits['train']}  val={splits['val']}  test={splits['test']}")
    print(f"  Output: {out_path}")
    sizes = sum((out_path / f).stat().st_size
                for f in ["rgb.dat", "dem.dat", "landuse.dat"]
                if (out_path / f).exists())
    print(f"  Cache size: {sizes / 1e9:.1f} GB")
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GeoTIFF crops into numpy memmaps")
    parser.add_argument("--json",   required=True, help="Path to final_clustered_samples.json")
    parser.add_argument("--base",   required=True, help="Base dataset directory (contains satellite_images/ etc.)")
    parser.add_argument("--out",    required=True, help="Output cache directory")
    parser.add_argument("--crop",   type=int,   default=224,     help="Crop size in pixels (default 224)")
    parser.add_argument("--jitter", type=int,   default=40,      help="Jitter padding in pixels (default 40)")
    parser.add_argument("--seed",   type=int,   default=42,      help="Random seed (must match training)")
    parser.add_argument("--clamp",  type=float, default=80_000., help="Target clamp max (default 80000)")
    args = parser.parse_args()

    preprocess(
        json_file=args.json,
        base_dir=args.base,
        out_dir=args.out,
        crop_size=args.crop,
        jitter_pixels=args.jitter,
        random_seed=args.seed,
        target_clamp_max=args.clamp,
    )
