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
from rasterio.warp import transform
import unicodedata

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

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
        
        with open(json_file, 'rb') as f:
            data = json.load(f)
        
        data = [s for s in data if s.get('population_15min_walk') is not None]
        
        if cities_filter is not None:
            data = [s for s in data if s['city'] in cities_filter]
        
        self.samples = data
        
        self.city_to_files = self._create_city_mapping()
        
        self.tiff_cache = {}
        if cache_images:
            for s in self.samples:
                city_key = self._get_clean_key(s['city'])
                if city_key not in self.tiff_cache and city_key in self.city_to_files:
                    try:
                        files = self.city_to_files[city_key]
                        self.tiff_cache[city_key] = {
                            'rgb': rasterio.open(files['rgb']),
                            'dem': rasterio.open(files['dem']),
                            'landuse': rasterio.open(files['landuse'])
                        }
                    except Exception:
                        pass

    def _get_clean_key(self, filename_or_city: str) -> str:
        s = str(filename_or_city).replace('.tif', '').replace('.tiff', '')
        suffixes = ['_LandUse', 'LandUse', '_DEM', 'DEM', '_10m', '10m']
        for suff in suffixes:
            s = s.replace(suff, '')
            
        s = s.lower()
        
        replacements = {
            'ø': 'o', 'å': 'a', 'æ': 'ae', 
            'ö': 'o', 'ü': 'u', 'ä': 'a',
            'é': 'e', 'è': 'e'
        }
        for char, repl in replacements.items():
            s = s.replace(char, repl)

        s = ''.join(c for c in unicodedata.normalize('NFD', s) 
                   if unicodedata.category(c) != 'Mn')
        
        s = ''.join(c for c in s if c.isalnum())
        
        return s

    def _create_city_mapping(self) -> Dict[str, Dict[str, Path]]:
        mapping = {}
        
        rgb_files = list(self.rgb_dir.glob('*.tif'))
        dem_files = list(self.dem_dir.glob('*.tif'))
        lu_files = list(self.landuse_dir.glob('*.tif'))
        
        dem_map = {self._get_clean_key(f.name): f for f in dem_files}
        lu_map = {self._get_clean_key(f.name): f for f in lu_files}
        
        for rgb_path in rgb_files:
            base_key = self._get_clean_key(rgb_path.name)
            
            if base_key in dem_map and base_key in lu_map:
                mapping[base_key] = {
                    'rgb': rgb_path,
                    'dem': dem_map[base_key],
                    'landuse': lu_map[base_key]
                }
                
        return mapping

    def _gps_to_pixel(self, raster, lon, lat):
        row, col = raster.index(lon, lat)
        return int(row), int(col)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        city_key = self._get_clean_key(sample['city'])
        
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
            return self._get_empty_sample()

        src_rgb = rasters['rgb']
        src_dem = rasters['dem']
        src_lu = rasters['landuse']
        
        if src_rgb.width != src_dem.width or src_rgb.height != src_dem.height:
           pass 
    
        try:
            row, col = self._gps_to_pixel(src_rgb, sample['lon'], sample['lat'])
        except Exception:
            return self._get_empty_sample()

        if self.augment:
            row += np.random.randint(-20, 21)
            col += np.random.randint(-20, 21)

        half = self.crop_size // 2
        r_start = row - half
        c_start = col - half
        
        r_start = max(0, min(r_start, src_rgb.height - self.crop_size))
        c_start = max(0, min(c_start, src_rgb.width - self.crop_size))
        
        window = Window(c_start, r_start, self.crop_size, self.crop_size)
        
        try:
            rgb = src_rgb.read([1,2,3], window=window, boundless=True, fill_value=0)
            dem = src_dem.read(1, window=window, boundless=True, fill_value=0)
            lu = src_lu.read(1, window=window, boundless=True, fill_value=0)
            
            rgb = rgb.astype(np.float32) / 255.0
            
            if dem.max() > dem.min():
                dem = (dem - dem.min()) / (dem.max() - dem.min())
            else:
                dem = np.zeros_like(dem, dtype=np.float32)
            
            lu_onehot = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
            for k in range(8):
                lu_onehot[k] = (lu == k)
            
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
                'metadata': {'city': sample['city'], 'idx': idx, 'cluster': sample.get('cluster', 0)}
            }
            
        except Exception:
            return self._get_empty_sample()

    def _get_empty_sample(self):
        return {
            'image': torch.zeros((12, self.crop_size, self.crop_size), dtype=torch.float32),
            'target': torch.tensor([0.0], dtype=torch.float32),
            'metadata': {
                'city': "unknown_missing_file", 
                'idx': -1,
                'cluster': -1,
                'lat': 0.0,
                'lon': 0.0
            }
        }

def create_city_split(
    json_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    city_counts = {}
    for sample in data:
        city = sample['city']
        city_counts[city] = city_counts.get(city, 0) + 1
    
    cities = list(city_counts.keys())
    
    np.random.seed(random_seed)
    np.random.shuffle(cities)
    
    n_cities = len(cities)
    n_train = int(n_cities * train_ratio)
    n_val = int(n_cities * val_ratio)
    
    train_cities = cities[:n_train]
    val_cities = cities[n_train:n_train + n_val]
    test_cities = cities[n_train + n_val:]
    
    return train_cities, val_cities, test_cities

def get_dataloaders(
    json_file: str,
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_tabular: bool = False,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_cities, val_cities, test_cities = create_city_split(
        json_file, train_ratio, val_ratio, random_seed=random_seed
    )
    
    class MultiChannelAugmentation:
        def __init__(self, augment=False):
            self.augment = augment
        
        def __call__(self, x):
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
    
    train_transform = MultiChannelAugmentation(augment=True)
    val_test_transform = MultiChannelAugmentation(augment=False)
    
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    BASE_DIR = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset"
    JSON_FILE = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset\final_clustered_samples.json"
    
    train_cities, val_cities, test_cities = create_city_split(JSON_FILE)
    
    dataset = GeospatialPopulationDataset(
        json_file=JSON_FILE,
        base_dir=BASE_DIR,
        crop_size=224,
        transform=None,
        use_tabular_features=True,
        cities_filter=train_cities[:3],
        cache_images=True
    )
    
    sample = dataset[0]
    
    rgb = sample['image'][:3].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)
    
    dem = sample['image'][3].numpy()
    
    landuse = sample['image'][4:12].sum(dim=0).numpy()
    
    plt.show()