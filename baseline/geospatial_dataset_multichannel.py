"""
GEOSPATIAL PYTORCH DATASET - MULTI-CHANNEL VERSION (SEPARATE FOLDERS)
Dataset to extract 224x224 crops from TIFF satellite images
with RGB (3) + DEM Height (1) + Land Use One-Hot (8) = 12 channels

FOLDER STRUCTURE:
PopulationDataset/
├── satellite_images/         ← RGB files
├── dem_height/              ← DEM files
├── segmentation_land_use/   ← Land Use files
└── final_clustered_samples.json
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

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import rasterio
from rasterio.windows import Window
import unicodedata
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

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
        
        # 1. LOAD JSON
        print(f"📂 Loading JSON from {json_file}...")
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        # Filter valid labels
        data = [s for s in data if s.get('population_15min_walk') is not None]
        
        if cities_filter is not None:
            data = [s for s in data if s['city'] in cities_filter]
        
        self.samples = data
        
        # 2. CREATE ROBUST MAPPING
        self.city_to_files = self._create_city_mapping()
        
        # 3. PRE-LOAD TIFF IMAGES
        self.tiff_cache = {}
        if cache_images:
            print(f"💾 Pre-loading TIFF images...")
            for s in self.samples:
                city_key = self._get_clean_key(s['city']) # Usa la chiave pulita anche qui
                if city_key not in self.tiff_cache and city_key in self.city_to_files:
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
                missing = []
                if base_key not in dem_map: missing.append("DEM")
                if base_key not in lu_map: missing.append("LandUse")
                # print(f"⚠️ Missing {missing} for city key: {base_key} (File: {rgb_path.name})")
                missing_count += 1
                
        print(f"✅ Mapped {len(mapping)} cities completely. ({missing_count} incomplete ignored)")
        return mapping

    def _gps_to_pixel(self, raster, lon, lat):
        # Usa la trasformazione affine del raster
        row, col = raster.index(lon, lat)
        return int(row), int(col)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Usa la chiave pulita
        city_key = self._get_clean_key(sample['city'])
        
        # Recupera rasters
        if self.cache_images and city_key in self.tiff_cache:
            rasters = self.tiff_cache[city_key]
        elif city_key in self.city_to_files:
            files = self.city_to_files[city_key]
            rasters = {
                'rgb': rasterio.open(files['rgb']),
                'dem': rasterio.open(files['dem']),
                'landuse': rasterio.open(files['landuse'])
            }
        else:
            # Fallback per sicurezza, ma non dovrebbe succedere se filtrati bene
            # Restituisce un tensore vuoto invece di crashare
            print(f"❌ Missing files for {sample['city']} (key: {city_key})")
            return self._get_empty_sample()

        # RGB è il master per le coordinate
        src_rgb = rasters['rgb']
        src_dem = rasters['dem']
        src_lu = rasters['landuse']
        
        # CHECK VELOCE DIMENSIONI (Per evitare il problema geometrico)
        if src_rgb.width != src_dem.width or src_rgb.height != src_dem.height:
           # Se le dimensioni sono diverse, le coordinate pixel non combaciano!
           # Per ora stampiamo un warning, in futuro serve la riproiezione.
           pass 

        # Calcola pixel centrali
        try:
            row, col = self._gps_to_pixel(src_rgb, sample['lon'], sample['lat'])
        except Exception:
            return self._get_empty_sample()

        # Data augmentation (jitter)
        if self.augment:
            row += np.random.randint(-20, 21)
            col += np.random.randint(-20, 21)

        # Estrazione crop
        half = self.crop_size // 2
        r_start = row - half
        c_start = col - half
        
        # Gestione bordi (clipping)
        # Nota: assumiamo che tutti i raster abbiano stessa dimensione per ora
        r_start = max(0, min(r_start, src_rgb.height - self.crop_size))
        c_start = max(0, min(c_start, src_rgb.width - self.crop_size))
        
        window = Window(c_start, r_start, self.crop_size, self.crop_size)
        
        try:
            # Leggi i dati (gestendo le differenze di canali)
            rgb = src_rgb.read([1,2,3], window=window, boundless=True, fill_value=0)
            dem = src_dem.read(1, window=window, boundless=True, fill_value=0)
            lu = src_lu.read(1, window=window, boundless=True, fill_value=0)
            
            # Normalizzazione
            rgb = rgb.astype(np.float32) / 255.0
            
            # DEM Normalization (Standardizzazione base)
            # Evitiamo divisioni per zero se il crop è piatto
            if dem.max() > dem.min():
                dem = (dem - dem.min()) / (dem.max() - dem.min())
            else:
                dem = np.zeros_like(dem, dtype=np.float32)
            
            # Land Use One-Hot (8 classi)
            lu_onehot = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
            for k in range(8):
                lu_onehot[k] = (lu == k)
            
            # Assembla tensore: (3 RGB + 1 DEM + 8 LU) = 12 channels
            # Shape finale: (12, 224, 224)
            combined = np.concatenate([
                rgb, 
                dem[np.newaxis, ...], 
                lu_onehot
            ], axis=0)
            
            tensor = torch.from_numpy(combined).float()
            
            if self.transform:
                tensor = self.transform(tensor)
                
            return {
                'image': tensor,
                'target': torch.tensor([sample['population_15min_walk']], dtype=torch.float32),
                'metadata': {'city': sample['city'], 'idx': idx, 'cluster': sample.get('cluster', 0), 'lat': sample.get('lat', 0.0), 'lon': sample.get('lon', 0.0)}
            }
            
        except Exception as e:
            print(f"Read error: {e}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        return {
            'image': torch.zeros((12, self.crop_size, self.crop_size), dtype=torch.float32),
            'target': torch.tensor([0.0], dtype=torch.float32),
            'metadata': {
                'city': "unknown_missing_file", 
                'idx': -1,
                'cluster': -1,  # Default cluster for missing samples
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
    base_dir: str,  # Directory containing the 3 subdirectories
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_tabular: bool = False,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation and test with geographic split.
    
    Args:
        json_file: Path to JSON with samples
        base_dir: Base directory containing:
                  - satellite_images/
                  - dem_height/
                  - segmentation_land_use/
    
    Returns:
        (train_loader, val_loader, test_loader)
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
        cache_images=True
    )
    
    val_dataset = GeospatialPopulationDataset(
        json_file=json_file,
        base_dir=base_dir,
        transform=val_test_transform,
        use_tabular_features=use_tabular,
        cities_filter=val_cities,
        augment=False,
        cache_images=True
    )
    
    test_dataset = GeospatialPopulationDataset(
        json_file=json_file,
        base_dir=base_dir,
        transform=val_test_transform,
        use_tabular_features=use_tabular,
        cities_filter=test_cities,
        augment=False,
        cache_images=True
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
    
    return train_loader, val_loader, test_loader


# ============================================================================
# DATASET TEST
# ============================================================================

if __name__ == "__main__":
    """
    Quick dataset test to verify everything works
    """
    import matplotlib.pyplot as plt
    
    # Paths (MODIFY WITH YOURS)
    BASE_DIR = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset"
    JSON_FILE = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset\final_clustered_samples.json"
    
    print("=" * 80)
    print(" MULTI-CHANNEL DATASET TEST (SEPARATE FOLDERS)")
    print("=" * 80)
    
    # 1. Test geographic split
    train_cities, val_cities, test_cities = create_city_split(JSON_FILE)
    
    # 2. Create test dataset
    print("\n" + "=" * 80)
    print(" Creating test dataset...")
    print("=" * 80)
    
    dataset = GeospatialPopulationDataset(
        json_file=JSON_FILE,
        base_dir=BASE_DIR,
        crop_size=224,
        transform=None,
        use_tabular_features=True,
        cities_filter=train_cities[:3],  # Only first 3 cities for fast test
        cache_images=True
    )
    
    print(f"\n✅ Dataset created with {len(dataset)} samples")
    
    # 3. Test loading samples
    print("\n" + "=" * 80)
    print(" Testing sample loading...")
    print("=" * 80)
    
    sample = dataset[0]
    
    print(f"Sample 0:")
    print(f"  Image shape: {sample['image'].shape}")  # Should be (12, 224, 224)
    print(f"  Target: {sample['target'].item():.2f} people")
    print(f"  Metadata: {sample['metadata']}")
    
    # Visualize RGB channels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB visualization
    rgb = sample['image'][:3].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Channels (0-2)')
    axes[0].axis('off')
    
    # DEM visualization
    dem = sample['image'][3].numpy()
    axes[1].imshow(dem, cmap='terrain')
    axes[1].set_title('DEM Height (Channel 3)')
    axes[1].axis('off')
    
    # Land Use visualization (sum of all classes)
    landuse = sample['image'][4:12].sum(dim=0).numpy()
    axes[2].imshow(landuse, cmap='tab10')
    axes[2].set_title('Land Use (Channels 4-11)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('multichannel_test.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved: multichannel_test.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED!")
    print("=" * 80)