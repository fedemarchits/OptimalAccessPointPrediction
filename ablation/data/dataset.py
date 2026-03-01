"""
Geospatial population dataset — multi-channel version.

Loads 224×224 crops from city-level GeoTIFF files and assembles a
12-channel tensor: RGB (3) + DEM height (1) + LandUse one-hot (8).

Key design decisions (ported from model2 with improvements):
  - Coordinates taken from start_point['lng'/'lat'] (precise walk origin)
  - DEM normalised with global constants [-50 m, 3000 m] — avoids
    per-crop leakage and keeps val/test normalisation consistent
  - Targets clamped to [0, 80_000] then log1p-normalised
  - Augmentation jitter configurable (default ±40 px)
  - `tabular` key always present in every batch (zeros when disabled),
    so the DataLoader collation is identical for both model types
"""

from __future__ import annotations

import json
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import rasterio
from rasterio.windows import Window

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Number of OSM tabular features expected in the JSON (osm_features key)
N_TABULAR_FEATURES: int = 15

# Global DEM normalisation range (metres)
_DEM_MIN: float = -50.0
_DEM_MAX: float = 3_000.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GeospatialPopulationDataset(Dataset):

    def __init__(
        self,
        json_file: str,
        base_dir: str,
        crop_size: int = 224,
        transform=None,
        use_tabular_features: bool = False,
        cache_images: bool = True,
        cities_filter: Optional[List[str]] = None,
        augment: bool = False,
        normalize_targets: bool = True,
        jitter_pixels: int = 40,
        target_clamp_max: float = 80_000.0,
    ):
        self.base_dir   = Path(base_dir)
        self.rgb_dir    = self.base_dir / "satellite_images"
        self.dem_dir    = self.base_dir / "dem_height"
        self.landuse_dir = self.base_dir / "segmentation_land_use"

        self.crop_size           = crop_size
        self.transform           = transform
        self.use_tabular_features = use_tabular_features
        self.cache_images        = cache_images
        self.augment             = augment
        self.normalize_targets   = normalize_targets
        self.jitter_pixels       = jitter_pixels
        self.target_clamp_max    = target_clamp_max

        # 1. Load JSON
        print(f"Loading JSON: {json_file}")
        with open(json_file, "rb") as f:
            data = json.load(f)

        data = [s for s in data if s.get("population_15min_walk") is not None]
        if cities_filter is not None:
            data = [s for s in data if s["city"] in cities_filter]

        # 2. Build city → file mapping
        self.city_to_files = self._build_city_mapping()

        # 3. Drop samples whose city files are missing
        before = len(data)
        data = [s for s in data if self._clean_key(s["city"]) in self.city_to_files]
        dropped = before - len(data)
        if dropped:
            print(f"  Dropped {dropped} samples with missing files")

        self.samples = data

        # 4. Target normalisation statistics (computed on this split)
        if self.normalize_targets:
            raw = np.array([s["population_15min_walk"] for s in self.samples])
            raw_clamped = np.clip(raw, 0.0, self.target_clamp_max)
            log_vals = np.log1p(raw_clamped)
            self.log_mean = float(log_vals.mean())
            self.log_std  = float(log_vals.std())
            print(f"  Targets (log1p): mean={self.log_mean:.3f}, std={self.log_std:.3f}")

        # 5. Tabular features
        self.tabular_data: Optional[np.ndarray] = None
        self.tabular_mean: Optional[np.ndarray] = None
        self.tabular_std:  Optional[np.ndarray] = None

        if self.use_tabular_features:
            raw_tab = []
            for s in self.samples:
                feats = s.get("osm_features")
                if feats is not None and len(feats) == N_TABULAR_FEATURES:
                    raw_tab.append(feats)
                else:
                    raw_tab.append([0.0] * N_TABULAR_FEATURES)
            log_arr = np.log1p(np.array(raw_tab, dtype=np.float32))
            self.tabular_data = log_arr
            self.tabular_mean = log_arr.mean(axis=0)
            self.tabular_std  = log_arr.std(axis=0) + 1e-8
            n_nonzero = (np.array(raw_tab) > 0).any(axis=1).sum()
            print(f"  Tabular: {n_nonzero}/{len(self.samples)} samples with non-zero features")

        # 6. Pre-load rasters
        self.tiff_cache: Dict = {}
        if cache_images:
            print(f"  Pre-loading TIFF images…")
            for city_key in {self._clean_key(s["city"]) for s in self.samples}:
                if city_key in self.city_to_files:
                    try:
                        f = self.city_to_files[city_key]
                        self.tiff_cache[city_key] = {
                            "rgb":     rasterio.open(f["rgb"]),
                            "dem":     rasterio.open(f["dem"]),
                            "landuse": rasterio.open(f["landuse"]),
                        }
                    except Exception as e:
                        print(f"    Warning: could not open {city_key}: {e}")
            print(f"  Cached {len(self.tiff_cache)} cities")

    # ── Key normalisation ─────────────────────────────────────────────────

    @staticmethod
    def _clean_key(s: str) -> str:
        """Normalise a city name or filename to a lowercase alphanumeric key."""
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

    # ── File mapping ──────────────────────────────────────────────────────

    def _build_city_mapping(self) -> Dict[str, Dict[str, Path]]:
        rgb_files = list(self.rgb_dir.glob("*.tif"))
        dem_files = list(self.dem_dir.glob("*.tif"))
        lu_files  = list(self.landuse_dir.glob("*.tif"))

        dem_map = {self._clean_key(f.name): f for f in dem_files}
        lu_map  = {self._clean_key(f.name): f for f in lu_files}

        print(
            f"  File index: {len(rgb_files)} RGB, "
            f"{len(dem_files)} DEM, {len(lu_files)} LandUse"
        )
        mapping, missing = {}, 0
        for rgb_path in rgb_files:
            key = self._clean_key(rgb_path.name)
            if key in dem_map and key in lu_map:
                mapping[key] = {"rgb": rgb_path, "dem": dem_map[key], "landuse": lu_map[key]}
            else:
                missing += 1
        print(f"  Mapped {len(mapping)} complete cities ({missing} incomplete skipped)")
        return mapping

    # ── Normalisation helpers ─────────────────────────────────────────────

    def denormalize_target(self, t: torch.Tensor) -> torch.Tensor:
        """Invert log1p normalisation: expm1(t) = exp(t) - 1."""
        if not self.normalize_targets:
            return t
        return torch.expm1(t)

    def set_tabular_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Override tabular normalisation stats with those from the training split.
        Must be called on val/test datasets before use.
        """
        self.tabular_mean = mean
        self.tabular_std  = std

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample   = self.samples[idx]
        city_key = self._clean_key(sample["city"])

        # Resolve rasters from cache or disk
        if self.cache_images and city_key in self.tiff_cache:
            rasters = self.tiff_cache[city_key]
        else:
            f = self.city_to_files[city_key]
            rasters = {
                "rgb":     rasterio.open(f["rgb"]),
                "dem":     rasterio.open(f["dem"]),
                "landuse": rasterio.open(f["landuse"]),
            }

        src_rgb = rasters["rgb"]
        src_dem = rasters["dem"]
        src_lu  = rasters["landuse"]

        # Coordinates — prefer start_point over top-level lat/lon
        sp   = sample.get("start_point", {})
        lat  = sp.get("lat",  sample.get("lat",  0.0))
        lon  = sp.get("lng",  sample.get("lon",  0.0))

        # GPS → pixel
        try:
            row, col = src_rgb.index(lon, lat)
        except Exception:
            row, col = src_rgb.height // 2, src_rgb.width // 2

        # Spatial jitter (training only)
        if self.augment and self.jitter_pixels > 0:
            row += np.random.randint(-self.jitter_pixels, self.jitter_pixels + 1)
            col += np.random.randint(-self.jitter_pixels, self.jitter_pixels + 1)

        # Crop window (clamped to raster extent)
        half    = self.crop_size // 2
        r_start = int(np.clip(row - half, 0, src_rgb.height - self.crop_size))
        c_start = int(np.clip(col - half, 0, src_rgb.width  - self.crop_size))
        window  = Window(c_start, r_start, self.crop_size, self.crop_size)

        try:
            # ── Image channels ───────────────────────────────────────────
            rgb = src_rgb.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            dem = src_dem.read(1,          window=window, boundless=True, fill_value=0)
            lu  = src_lu.read(1,           window=window, boundless=True, fill_value=0)

            # RGB: uint8 → [0, 1]
            rgb = rgb.astype(np.float32) / 255.0

            # DEM: global normalisation (no per-crop scaling)
            dem = np.clip(dem.astype(np.float32), _DEM_MIN, _DEM_MAX)
            dem = (dem - _DEM_MIN) / (_DEM_MAX - _DEM_MIN)

            # LandUse: one-hot encode 8 classes → (8, H, W)
            lu_onehot = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
            for k in range(8):
                lu_onehot[k] = (lu == k)

            # Stack all channels: (3 + 1 + 8) = 12
            tensor = torch.from_numpy(
                np.concatenate([rgb, dem[np.newaxis], lu_onehot], axis=0)
            ).float()

            if self.transform:
                tensor = self.transform(tensor)

            # ── Target ───────────────────────────────────────────────────
            raw_target = float(np.clip(sample["population_15min_walk"], 0.0, self.target_clamp_max))
            norm_target = float(np.log1p(raw_target)) if self.normalize_targets else raw_target

            # ── Tabular ──────────────────────────────────────────────────
            # Always present in every batch (zeros when tabular is disabled)
            if self.use_tabular_features and self.tabular_data is not None:
                log_feats  = self.tabular_data[idx]
                norm_feats = (log_feats - self.tabular_mean) / self.tabular_std
                tabular    = torch.from_numpy(norm_feats).float()
            else:
                tabular = torch.zeros(N_TABULAR_FEATURES, dtype=torch.float32)

            return {
                "image":      tensor,
                "tabular":    tabular,
                "target":     torch.tensor([norm_target], dtype=torch.float32),
                "target_raw": torch.tensor([raw_target],  dtype=torch.float32),
                "lat":        lat,
                "lon":        lon,
                "metadata": {
                    "city":    sample["city"],
                    "idx":     idx,
                    "cluster": sample.get("cluster", 0),
                    "lat":     lat,
                    "lon":     lon,
                },
            }

        except Exception as e:
            print(f"  Read error sample {idx} ({sample.get('city', '?')}): {e}")
            return {
                "image":      torch.zeros((12, self.crop_size, self.crop_size)),
                "tabular":    torch.zeros(N_TABULAR_FEATURES),
                "target":     torch.tensor([0.0]),
                "target_raw": torch.tensor([0.0]),
                "lat":        0.0,
                "lon":        0.0,
                "metadata": {
                    "city":    sample.get("city", "unknown"),
                    "idx":     idx,
                    "cluster": -1,
                    "lat":     0.0,
                    "lon":     0.0,
                },
            }


# ---------------------------------------------------------------------------
# Augmentation transform (multi-channel safe)
# ---------------------------------------------------------------------------

class MultiChannelAugmentation:
    """Random flips and 90-degree rotations for (12, H, W) tensors."""

    def __init__(self, augment: bool = False):
        self.augment = augment

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return x
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[2])
        if np.random.rand() > 0.5:
            x = torch.flip(x, dims=[1])
        k = np.random.randint(0, 4)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        return x


# ---------------------------------------------------------------------------
# Fast cached dataset (loads from preprocessed numpy memmaps)
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """
    Loads pre-extracted crops from numpy memmaps written by preprocess.py.

    Compared to GeospatialPopulationDataset this avoids all GeoTIFF I/O
    during training — every __getitem__ is a cheap array slice + a few
    numpy ops.  Supports the full spatial jitter augmentation because crops
    are stored at (crop_size + 2*jitter_pixels) px.

    Expected cache_dir layout (created by preprocess.py):
        rgb.dat        (N, 3, PAD, PAD)  uint8
        dem.dat        (N, 1, PAD, PAD)  float32  already normalised [0,1]
        landuse.dat    (N, 1, PAD, PAD)  uint8    class labels 0-7
        targets.npy    (N,)              float32  log1p population
        tabular.npy    (N, 15)           float32  z-scored OSM features
        index.json     [{city, cluster, split, valid}, ...]
        stats.json     normalisation constants
    """

    def __init__(
        self,
        cache_dir: str,
        split: str,                      # "train" | "val" | "test"
        crop_size: int = 224,
        use_tabular: bool = False,
        augment: bool = False,
    ):
        self.crop_size   = crop_size
        self.use_tabular = use_tabular
        self.augment     = augment

        cache = Path(cache_dir)

        with open(cache / "stats.json") as f:
            stats = json.load(f)
        self.log_mean      = stats["log_mean"]
        self.log_std       = stats["log_std"]
        self.pad_size      = stats["pad_size"]
        self.jitter_pixels = stats["jitter_pixels"]

        with open(cache / "index.json") as f:
            index = json.load(f)

        # Build country → integer label mapping (from ALL entries, not just this split,
        # so the mapping is consistent across train/val/test datasets)
        all_countries = sorted({e["city"].split(", ")[-1] for e in index})
        self.country_to_idx = {c: i for i, c in enumerate(all_countries)}
        self.n_countries    = len(all_countries)

        # Filter to the requested split and valid samples only
        self._indices = [
            i for i, e in enumerate(index)
            if e["split"] == split and e["valid"]
        ]
        self._meta = [index[i] for i in self._indices]

        print(f"  CachedDataset [{split}]: {len(self._indices)} samples  "
              f"({self.n_countries} countries)")

        # Use memory-mapped files — reads only the requested slices from disk
        self._rgb     = np.lib.format.open_memmap(cache / "rgb.dat",     mode="r")
        self._dem     = np.lib.format.open_memmap(cache / "dem.dat",     mode="r")
        self._landuse = np.lib.format.open_memmap(cache / "landuse.dat", mode="r")
        self._targets = np.load(cache / "targets.npy")
        self._tabular = np.load(cache / "tabular.npy")

        # Per-sample city z-score stats for denormalisation
        self._city_log_mean = np.load(cache / "city_log_mean.npy")
        self._city_log_std  = np.load(cache / "city_log_std.npy")

        # Per-city RGB normalisation (generated by compute_rgb_stats.py)
        # If absent (older caches) the dataset falls back to raw [0, 1] RGB
        # and the backbone applies ImageNet normalisation as before.
        _rgb_mean_path = cache / "rgb_city_mean.npy"
        _rgb_std_path  = cache / "rgb_city_std.npy"
        if _rgb_mean_path.exists() and _rgb_std_path.exists():
            self._rgb_city_mean = np.load(_rgb_mean_path)   # (N, 3)
            self._rgb_city_std  = np.load(_rgb_std_path)    # (N, 3)
            print(f"  Per-city RGB normalisation: enabled")
        else:
            self._rgb_city_mean = None
            self._rgb_city_std  = None

    def denormalize_target(self, t: torch.Tensor) -> torch.Tensor:
        """Fallback: identity (z-score cannot be inverted without city stats)."""
        return t

    def make_denorm_fn(self, city_log_means: np.ndarray, city_log_stds: np.ndarray):
        """
        Return a closure that converts per-city z-scores back to absolute population.
        city_log_means / city_log_stds: 1-D arrays aligned with the prediction array.
        """
        means = torch.from_numpy(city_log_means.astype(np.float32))
        stds  = torch.from_numpy(city_log_stds.astype(np.float32))
        def _denorm(t: torch.Tensor) -> torch.Tensor:
            return torch.expm1(t * stds + means)
        return _denorm

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        src_idx = self._indices[idx]
        meta    = self._meta[idx]

        # Spatial jitter: random 224×224 sub-crop from the padded patch
        if self.augment and self.jitter_pixels > 0:
            max_offset = 2 * self.jitter_pixels
            r0 = np.random.randint(0, max_offset + 1)
            c0 = np.random.randint(0, max_offset + 1)
        else:
            r0 = self.jitter_pixels  # centre crop
            c0 = self.jitter_pixels

        rs = slice(r0, r0 + self.crop_size)
        cs = slice(c0, c0 + self.crop_size)

        rgb     = self._rgb[src_idx, :, rs, cs].astype(np.float32) / 255.0
        dem     = self._dem[src_idx, :, rs, cs].copy()
        lu_raw  = self._landuse[src_idx, 0, rs, cs]

        # One-hot encode land use
        lu = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
        for k in range(8):
            lu[k] = (lu_raw == k)

        tensor = torch.from_numpy(
            np.concatenate([rgb, dem, lu], axis=0)
        ).float()

        # Flip / rotate augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if np.random.rand() > 0.5:
                tensor = torch.flip(tensor, dims=[1])
            k = np.random.randint(0, 4)
            if k > 0:
                tensor = torch.rot90(tensor, k, dims=[1, 2])

            # Color jitter on RGB channels only (channels 0-2).
            # This prevents the backbone from memorizing country-specific
            # color distributions (roof colors, pavement tones, vegetation
            # hues), which are the main source of city-level shortcut learning.
            rgb = tensor[:3]  # (3, H, W) in [0, 1]
            brightness_factor  = float(np.random.uniform(0.6, 1.4))
            contrast_factor    = float(np.random.uniform(0.7, 1.3))
            saturation_factor  = float(np.random.uniform(0.7, 1.3))
            hue_factor         = float(np.random.uniform(-0.15, 0.15))
            rgb = TF.adjust_brightness(rgb, brightness_factor)
            rgb = TF.adjust_contrast(rgb, contrast_factor)
            rgb = TF.adjust_saturation(rgb, saturation_factor)
            rgb = TF.adjust_hue(rgb, hue_factor)
            rgb = rgb.clamp(0.0, 1.0)
            tensor = torch.cat([rgb, tensor[3:]], dim=0)

        # Per-city RGB normalisation.
        # The backbone will still apply its standard ImageNet normalisation,
        # so we pre-warp the values so that operation reduces to a per-city
        # z-score.  Derivation:
        #   backbone does: out = (x - imagenet_m) / imagenet_s
        #   we want: out  = (rgb - city_m) / city_s
        #   so feed: x   = imagenet_m + imagenet_s * (rgb - city_m) / city_s
        if self._rgb_city_mean is not None:
            city_m = torch.tensor(
                self._rgb_city_mean[src_idx], dtype=torch.float32).view(3, 1, 1)
            city_s = torch.tensor(
                self._rgb_city_std[src_idx],  dtype=torch.float32).view(3, 1, 1)
            _imn_m = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            _imn_s = torch.tensor(
                [0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            tensor = torch.cat([
                _imn_m + _imn_s * (tensor[:3] - city_m) / city_s,
                tensor[3:]
            ], dim=0)

        target         = float(self._targets[src_idx])
        city_log_mean  = float(self._city_log_mean[src_idx])
        city_log_std   = float(self._city_log_std[src_idx])
        tabular = (torch.from_numpy(self._tabular[src_idx].copy()).float()
                   if self.use_tabular
                   else torch.zeros(N_TABULAR_FEATURES, dtype=torch.float32))

        country       = meta["city"].split(", ")[-1]
        country_label = self.country_to_idx.get(country, 0)

        return {
            "image":          tensor,
            "tabular":        tabular,
            "target":         torch.tensor([target],         dtype=torch.float32),
            "target_raw":     torch.tensor([float(np.expm1(city_log_std * target + city_log_mean))],
                                           dtype=torch.float32),
            "city_log_mean":  torch.tensor([city_log_mean],  dtype=torch.float32),
            "city_log_std":   torch.tensor([city_log_std],   dtype=torch.float32),
            "country_label":  torch.tensor(country_label,    dtype=torch.long),
            "lat":           0.0,
            "lon":           0.0,
            "metadata": {
                "city":    meta["city"],
                "country": country,
                "idx":     idx,
                "cluster": meta.get("cluster", 0),
                "lat":     0.0,
                "lon":     0.0,
            },
        }


def get_cached_dataloaders(
    cache_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_tabular: bool = False,
    crop_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, "CachedDataset"]:
    """
    Build DataLoaders from a preprocessed cache (created by preprocess.py).
    Drop-in replacement for get_dataloaders — returns the same 4-tuple.
    Supports num_workers > 0 safely (no rasterio handles to fork).
    """
    train_ds = CachedDataset(cache_dir, "train", crop_size, use_tabular, augment=True)
    val_ds   = CachedDataset(cache_dir, "val",   crop_size, use_tabular, augment=False)
    test_ds  = CachedDataset(cache_dir, "test",  crop_size, use_tabular, augment=False)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            drop_last=shuffle,   # drop last batch in train to avoid batch_size=1 with BatchNorm1d
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        train_ds,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

_SPLIT_FILE = Path(__file__).parent / "city_split.json"


def _city_split(
    json_file: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    # Use the canonical fixed split if it has been generated
    if _SPLIT_FILE.exists():
        with open(_SPLIT_FILE) as f:
            sp = json.load(f)
        print(f"Loaded fixed city split from {_SPLIT_FILE}")
        train = sp["train"]
        val   = sp["val"]
        test  = sp["test"]
    else:
        print("WARNING: city_split.json not found — computing split dynamically.")
        print("Run: python data/generate_split.py --json <path> to fix it.")
        with open(json_file, "r") as f:
            data = json.load(f)
        city_counts: Dict[str, int] = {}
        for s in data:
            city_counts[s["city"]] = city_counts.get(s["city"], 0) + 1
        cities = sorted(city_counts.keys())
        rng = np.random.default_rng(random_seed)
        rng.shuffle(cities)
        n      = len(cities)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)
        train  = cities[:n_tr]
        val    = cities[n_tr : n_tr + n_val]
        test   = cities[n_tr + n_val :]
    samples = lambda cs: sum(city_counts[c] for c in cs)
    print(
        f"\nCity split (seed={random_seed}): "
        f"train={len(train)} ({samples(train)} samples)  "
        f"val={len(val)} ({samples(val)} samples)  "
        f"test={len(test)} ({samples(test)} samples)"
    )
    return train, val, test


def get_dataloaders(
    json_file: str,
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_tabular: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    normalize_targets: bool = True,
    jitter_pixels: int = 40,
    target_clamp_max: float = 80_000.0,
    crop_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, GeospatialPopulationDataset]:
    """
    Build train / val / test DataLoaders from a single JSON file.

    Returns (train_loader, val_loader, test_loader, train_dataset).
    The train_dataset is returned so callers can access denormalize_target()
    and tabular normalisation statistics.
    """
    train_cities, val_cities, test_cities = _city_split(
        json_file, train_ratio, val_ratio, random_seed
    )

    train_tf = MultiChannelAugmentation(augment=True)
    eval_tf  = MultiChannelAugmentation(augment=False)

    def _make(cities, augment, transform):
        return GeospatialPopulationDataset(
            json_file=json_file,
            base_dir=base_dir,
            crop_size=crop_size,
            transform=transform,
            use_tabular_features=use_tabular,
            cache_images=True,
            cities_filter=cities,
            augment=augment,
            normalize_targets=normalize_targets,
            jitter_pixels=jitter_pixels if augment else 0,
            target_clamp_max=target_clamp_max,
        )

    train_ds = _make(train_cities, augment=True,  transform=train_tf)
    val_ds   = _make(val_cities,   augment=False, transform=eval_tf)
    test_ds  = _make(test_cities,  augment=False, transform=eval_tf)

    # Propagate train tabular stats to val/test so normalisation is consistent
    if use_tabular and train_ds.tabular_mean is not None:
        val_ds.set_tabular_stats(train_ds.tabular_mean, train_ds.tabular_std)
        test_ds.set_tabular_stats(train_ds.tabular_mean, train_ds.tabular_std)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        train_ds,
    )
