"""
GEOSPATIAL PYTORCH DATASET - MULTI-CHANNEL VERSION (SEPARATE FOLDERS) - FIXED
Dataset to extract 224x224 crops from TIFF satellite images
with RGB (3) + DEM Height (1) + Land Use One-Hot (8) = 12 channels

✅ FIXED: Added Log1p normalization for targets
✅ Target normalization strategy: log1p (handles 0 values, reduces scale)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import rasterio
from rasterio.windows import Window
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
from rasterio.windows import Window
from rasterio.warp import transform
import unicodedata

def get_safe_crop(src, lon, lat, crop_size=224):
    row, col = src.index(lon, lat)
    
    half = crop_size // 2
    row_start = row - half
    col_start = col - half
    

    row_start = max(0, min(row_start, src.height - crop_size))
    col_start = max(0, min(col_start, src.width - crop_size))
    
    window = Window(col_start, row_start, crop_size, crop_size)
    
    return src.read(window=window, boundless=True, fill_value=0)

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
        normalize_targets: bool = True,  # ✅ NEW: Enable target normalization
    ):
        self.base_dir = Path(base_dir)
        self.rgb_dir = self.base_dir / 'satellite_images'
        self.dem_dir = self.base_dir / 'dem_height'
        self.landuse_dir = self.base_dir / 'segmentation_land_use'
        
        self.crop_size = crop_size
        self.transform = transform
        self.use_tabular_features = use_tabular_features
        self.cache_images = cache_images
        self.augment = augment
        self.normalize_targets = normalize_targets  # ✅ NEW
        
        # 1. LOAD JSON
        print(f"📂 Loading JSON from {json_file}...")
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        # Filter valid labels
        data = [s for s in data if s.get('population_15min_walk') is not None]
        
        if cities_filter is not None:
            data = [s for s in data if s['city'] in cities_filter]
        
        # 2. CREATE ROBUST MAPPING
        self.city_to_files = self._create_city_mapping()
        
        # 3. FILTER SAMPLES - Remove samples for cities without all files
        print(f"🔍 Filtering samples based on available files...")
        initial_count = len(data)
        data = [s for s in data if self._get_clean_key(s['city']) in self.city_to_files]
        filtered_count = initial_count - len(data)
        if filtered_count > 0:
            print(f"   ⚠️ Removed {filtered_count} samples with missing files")
        
        self.samples = data
        
        # ✅ NEW: Compute normalization statistics
        if self.normalize_targets:
            targets = np.array([s['population_15min_walk'] for s in self.samples])
            self.target_mean = targets.mean()
            self.target_std = targets.std()
            self.target_min = targets.min()
            self.target_max = targets.max()
            
            print(f"\n📊 Target Statistics (RAW):")
            print(f"   Min: {self.target_min:.2f}")
            print(f"   Max: {self.target_max:.2f}")
            print(f"   Mean: {self.target_mean:.2f}")
            print(f"   Std: {self.target_std:.2f}")
            
            # Compute log1p statistics
            log_targets = np.log1p(targets)
            self.log_mean = log_targets.mean()
            self.log_std = log_targets.std()
            
            print(f"\n📊 Target Statistics (LOG1P):")
            print(f"   Min: {log_targets.min():.2f}")
            print(f"   Max: {log_targets.max():.2f}")
            print(f"   Mean: {self.log_mean:.2f}")
            print(f"   Std: {self.log_std:.2f}")
        
        # 4. PRE-LOAD TIFF IMAGES
        self.tiff_cache = {}
        if cache_images:
            print(f"💾 Pre-loading TIFF images...")
            unique_cities = set(self._get_clean_key(s['city']) for s in self.samples)
            for city_key in unique_cities:
                if city_key in self.city_to_files:
                    try:
                        files = self.city_to_files[city_key]
                        self.tiff_cache[city_key] = {
                            'rgb': rasterio.open(files['rgb']),
                            'dem': rasterio.open(files['dem']),
                            'landuse': rasterio.open(files['landuse'])
                        }
                    except Exception as e:
                        print(f"   ⚠️ Error loading {city_key}: {e}")
            print(f"   ✅ Loaded {len(self.tiff_cache)} cities into cache")

    def _get_clean_key(self, filename_or_city: str) -> str:

        s = str(filename_or_city).replace('.tif', '').replace('.tiff', '')
        suffixes = ['_LandUse', 'LandUse', '_DEM', 'DEM', '_10m', '10m']
        for suff in suffixes:
            s = s.replace(suff, '')
            
        s = s.lower()
        
        # 2. Conversione caratteri speciali manuale (CRITICO per Tromsø, Malmö, etc.)
        replacements = {
            'ø': 'o', 'å': 'a', 'æ': 'ae', 
            'ö': 'o', 'ü': 'u', 'ä': 'a',
            'é': 'e', 'è': 'e'
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)

        # 3. Normalizzazione standard (rimuove altri accenti)
        s = ''.join(c for c in unicodedata.normalize('NFD', s) 
                   if unicodedata.category(c) != 'Mn')
        
        # 4. Solo alfanumerici
        s = ''.join(c for c in s if c.isalnum())
        
        return s

    def _create_city_mapping(self) -> Dict[str, Dict[str, Path]]:
        mapping = {}
        
        # 1. Indicizza tutti i file disponibili
        rgb_files = list(self.rgb_dir.glob('*.tif'))
        dem_files = list(self.dem_dir.glob('*.tif'))
        lu_files = list(self.landuse_dir.glob('*.tif'))
        
        # Crea dizionari temporanei: clean_key -> full_path
        dem_map = {self._get_clean_key(f.name): f for f in dem_files}
        lu_map = {self._get_clean_key(f.name): f for f in lu_files}
        
        print(f"🔍 Indexing: found {len(rgb_files)} RGB, {len(dem_files)} DEM, {len(lu_files)} LandUse files.")
        
        missing_count = 0
        
        for rgb_path in rgb_files:
            # Chiave base dal file RGB (es: "bolognaitaly")
            base_key = self._get_clean_key(rgb_path.name)
            
            if base_key in dem_map and base_key in lu_map:
                mapping[base_key] = {
                    'rgb': rgb_path,
                    'dem': dem_map[base_key],
                    'landuse': lu_map[base_key]
                }
            else:
                missing_count += 1
                
        print(f"✅ Mapped {len(mapping)} cities completely. ({missing_count} incomplete ignored)")
        return mapping

    def _gps_to_pixel(self, raster, lon, lat):
        # Usa la trasformazione affine del raster
        row, col = raster.index(lon, lat)
        return int(row), int(col)

    def denormalize_target(self, normalized_target: torch.Tensor) -> torch.Tensor:
        """
        ✅ NEW: Denormalize log1p-normalized targets back to original scale
        
        Args:
            normalized_target: Log1p normalized target
        
        Returns:
            Original scale target
        """
        if not self.normalize_targets:
            return normalized_target
        
        # Reverse log1p: expm1(x) = exp(x) - 1
        return torch.expm1(normalized_target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Usa la chiave pulita
        city_key = self._get_clean_key(sample['city'])
        
        # Recupera rasters
        if self.cache_images and city_key in self.tiff_cache:
            rasters = self.tiff_cache[city_key]
        else:
            files = self.city_to_files[city_key]
            rasters = {
                'rgb': rasterio.open(files['rgb']),
                'dem': rasterio.open(files['dem']),
                'landuse': rasterio.open(files['landuse'])
            }

        # RGB è il master per le coordinate
        src_rgb = rasters['rgb']
        src_dem = rasters['dem']
        src_lu = rasters['landuse']

        # ✅ FIX: Recupera le coordinate corrette da 'start_point'
        # Struttura attesa: "start_point": { "lat": 44.493, "lng": 11.269 }
        start_point = sample.get('start_point', {})
        
        # Priorità a start_point['lng'], fallback su sample['lon'] se manca
        true_lat = start_point.get('lat', sample.get('lat', 0.0))
        true_lon = start_point.get('lng', sample.get('lon', 0.0))

        # Calcola pixel centrali usando le coordinate VERE
        try:
            row, col = self._gps_to_pixel(src_rgb, true_lon, true_lat)
        except Exception as e:
            print(f"⚠️ GPS conversion error for {sample['city']}: {e}")
            row, col = src_rgb.height // 2, src_rgb.width // 2

        # Data augmentation (jitter)
        if self.augment:
            row += np.random.randint(-40, 41)
            col += np.random.randint(-40, 41)

        # Estrazione crop
        half = self.crop_size // 2
        r_start = row - half
        c_start = col - half
        
        # Gestione bordi (clipping)
        r_start = max(0, min(r_start, src_rgb.height - self.crop_size))
        c_start = max(0, min(c_start, src_rgb.width - self.crop_size))
        
        window = Window(c_start, r_start, self.crop_size, self.crop_size)
        
        try:
            # Leggi i dati 
            rgb = src_rgb.read([1,2,3], window=window, boundless=True, fill_value=0)
            dem = src_dem.read(1, window=window, boundless=True, fill_value=0)
            lu = src_lu.read(1, window=window, boundless=True, fill_value=0)
            
            # Normalizzazione
            rgb = rgb.astype(np.float32) / 255.0
            
            DEM_MIN = -50.0
            DEM_MAX = 3000.0
            dem = np.clip(dem, DEM_MIN, DEM_MAX)
            dem = (dem - DEM_MIN) / (DEM_MAX - DEM_MIN)
            
            lu_onehot = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
            for k in range(8):
                lu_onehot[k] = (lu == k)
            
            combined = np.concatenate([rgb, dem[np.newaxis, ...], lu_onehot], axis=0)
            tensor = torch.from_numpy(combined).float()
            
            if self.transform:
                tensor = self.transform(tensor)
            
            # Target
            raw_target = sample['population_15min_walk']

            raw_target = np.clip(raw_target, a_min=0.0, a_max=80000.0) 

            if self.normalize_targets:
                # log1p normalization: log(1 + x)
                normalized_target = np.log1p(raw_target)
            else:
                normalized_target = raw_target
            
            # ✅ FIX: Restituisci le coordinate corrette nei metadati
            return {
                'image': tensor,
                'target': torch.tensor([normalized_target], dtype=torch.float32),
                'target_raw': torch.tensor([raw_target], dtype=torch.float32),
                'lat': true_lat,
                'lon': true_lon,
                'metadata': {
                    'city': sample['city'], 
                    'idx': idx, 
                    'cluster': sample.get('cluster', 0), 
                    'lat': true_lat,  # Coordinate precise del punto di partenza
                    'lon': true_lon   # Coordinate precise del punto di partenza
                }
            }
            
        except Exception as e:
            print(f"⚠️ Read error for sample {idx} ({sample['city']}): {e}")
            return {
                'image': torch.zeros((12, self.crop_size, self.crop_size), dtype=torch.float32),
                'target': torch.tensor([0.0], dtype=torch.float32),
                'target_raw': torch.tensor([0.0], dtype=torch.float32),
                'lat': 0.0,
                'lon': 0.0,
                'metadata': {
                    'city': sample.get('city', 'unknown'), 
                    'idx': idx,
                    'lat': 0.0,
                    'lon': 0.0
                }
            }


# ============================================================================
# HELPER FUNCTIONS FOR SPLITTING
# ============================================================================

def create_city_split(
    json_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits cities into train, validation and test sets.
    
    Returns:
        (train_cities, val_cities, test_cities)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get unique cities and count samples per city
    city_counts = {}
    for sample in data:
        city = sample['city']
        city_counts[city] = city_counts.get(city, 0) + 1
    
    cities = list(city_counts.keys())
    
    # Shuffle with seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(cities)
    
    # Calculate splits
    n_cities = len(cities)
    n_train = int(n_cities * train_ratio)
    n_val = int(n_cities * val_ratio)
    
    train_cities = cities[:n_train]
    val_cities = cities[n_train:n_train + n_val]
    test_cities = cities[n_train + n_val:]
    
    print(f"\n📊 Geographic Split:")
    print(f"  Train: {len(train_cities)} cities ({sum(city_counts[c] for c in train_cities)} samples)")
    print(f"  Val:   {len(val_cities)} cities ({sum(city_counts[c] for c in val_cities)} samples)")
    print(f"  Test:  {len(test_cities)} cities ({sum(city_counts[c] for c in test_cities)} samples)")
    
    return train_cities, val_cities, test_cities


def get_dataloaders(
    json_file: str,
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_tabular: bool = False,
    random_seed: int = 42,
    normalize_targets: bool = True  # ✅ NEW
) -> Tuple[DataLoader, DataLoader, DataLoader, GeospatialPopulationDataset]:
    """
    ✅ FIXED: Returns train dataset too for denormalization access
    """
    # 1. Geographic split of cities
    train_cities, val_cities, test_cities = create_city_split(
        json_file, train_ratio, val_ratio, random_seed=random_seed
    )
    
    # 2. Custom transforms for multi-channel data
    class MultiChannelAugmentation:
        """Custom augmentation for 12-channel tensors"""
        def __init__(self, augment=False):
            self.augment = augment
        
        def __call__(self, x):
            """x is (12, 224, 224) tensor"""
            if not self.augment:
                return x
            
            # Horizontal flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[2])
            
            # Vertical flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[1])
            
            # 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])
            
            return x
    
    train_transform = MultiChannelAugmentation(augment=True)
    val_test_transform = MultiChannelAugmentation(augment=False)
    
    # 3. Create datasets
    print("\n🏗️ Creating datasets...")
    
    train_dataset = GeospatialPopulationDataset(
        json_file=json_file,
        base_dir=base_dir,
        transform=train_transform,
        use_tabular_features=use_tabular,
        cities_filter=train_cities,
        augment=True,
        cache_images=True,
        normalize_targets=normalize_targets  # ✅ NEW
    )
    
    val_dataset = GeospatialPopulationDataset(
        json_file=json_file,
        base_dir=base_dir,
        transform=val_test_transform,
        use_tabular_features=use_tabular,
        cities_filter=val_cities,
        augment=False,
        cache_images=True,
        normalize_targets=normalize_targets  # ✅ NEW
    )
    
    test_dataset = GeospatialPopulationDataset(
        json_file=json_file,
        base_dir=base_dir,
        transform=val_test_transform,
        use_tabular_features=use_tabular,
        cities_filter=test_cities,
        augment=False,
        cache_images=True,
        normalize_targets=normalize_targets  # ✅ NEW
    )
    
    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset  # ✅ FIXED: Return dataset too
