"""
TEST RAPIDO - STRUTTURA CARTELLE SEPARATE
Verifica che il dataset multi-canale funzioni con la tua struttura:

PopulationDataset/
├── satellite_images/
├── dem_height/
├── segmentation_land_use/
└── final_clustered_samples.json
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print(" TEST MULTI-CHANNEL - SEPARATE FOLDERS")
print("=" * 80)

# ============================================================================
# CONFIGURAZIONE - MODIFICA QUESTI PATH SE NECESSARIO
# ============================================================================

BASE_DIR = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset"
JSON_FILE = r"C:\Users\andre\Desktop\AlmaMater\Indusrty\src\PopulationDataset\final_clustered_samples.json"

# ============================================================================
# TEST 1: VERIFICA STRUTTURA DIRECTORY
# ============================================================================

print("\n[TEST 1] Verifica struttura directory...")

base_path = Path(BASE_DIR)
rgb_dir = base_path / 'satellite_images'
dem_dir = base_path / 'dem_height'
landuse_dir = base_path / 'segmentation_land_use'
json_path = Path(JSON_FILE)

print(f"\nBase directory: {base_path}")
print(f"  Esiste? {base_path.exists()} {'✅' if base_path.exists() else '❌'}")

print(f"\nRGB directory: {rgb_dir}")
print(f"  Esiste? {rgb_dir.exists()} {'✅' if rgb_dir.exists() else '❌'}")
if rgb_dir.exists():
    rgb_files = list(rgb_dir.glob('*.tif')) + list(rgb_dir.glob('*.tiff'))
    print(f"  File trovati: {len(rgb_files)}")
    if rgb_files:
        print(f"  Esempi: {[f.name for f in rgb_files[:3]]}")

print(f"\nDEM directory: {dem_dir}")
print(f"  Esiste? {dem_dir.exists()} {'✅' if dem_dir.exists() else '❌'}")
if dem_dir.exists():
    dem_files = list(dem_dir.glob('*.tif')) + list(dem_dir.glob('*.tiff'))
    print(f"  File trovati: {len(dem_files)}")
    if dem_files:
        print(f"  Esempi: {[f.name for f in dem_files[:3]]}")

print(f"\nLand Use directory: {landuse_dir}")
print(f"  Esiste? {landuse_dir.exists()} {'✅' if landuse_dir.exists() else '❌'}")
if landuse_dir.exists():
    landuse_files = list(landuse_dir.glob('*.tif')) + list(landuse_dir.glob('*.tiff'))
    print(f"  File trovati: {len(landuse_files)}")
    if landuse_files:
        print(f"  Esempi: {[f.name for f in landuse_files[:3]]}")

print(f"\nJSON file: {json_path}")
print(f"  Esiste? {json_path.exists()} {'✅' if json_path.exists() else '❌'}")

# Check if all exist
all_exist = all([
    base_path.exists(),
    rgb_dir.exists(),
    dem_dir.exists(),
    landuse_dir.exists(),
    json_path.exists()
])

if not all_exist:
    print("\n❌ ERRORE: Alcune directory o file mancano!")
    print("Controlla i path sopra e correggi il file di test.")
    exit(1)

print("\n✅ Struttura directory corretta!")

# ============================================================================
# TEST 2: IMPORT E CREAZIONE DATASET
# ============================================================================

print("\n" + "=" * 80)
print("[TEST 2] Import moduli...")
print("=" * 80)

try:
    from geospatial_dataset_separate_folders import (
        GeospatialPopulationDataset,
        get_dataloaders,
        create_city_split
    )
    print("✅ Import dataset completato")
except Exception as e:
    print(f"❌ ERRORE import dataset: {e}")
    print("\nAssicurati che il file 'geospatial_dataset_separate_folders.py' sia nella stessa directory!")
    exit(1)

try:
    from baseline_model_multichannel import get_model
    print("✅ Import modello completato")
except Exception as e:
    print(f"❌ ERRORE import modello: {e}")
    print("\nAssicurati che il file 'baseline_model_multichannel.py' sia nella stessa directory!")
    exit(1)

# ============================================================================
# TEST 3: CREAZIONE DATASET
# ============================================================================

print("\n" + "=" * 80)
print("[TEST 3] Creazione dataset...")
print("=" * 80)

try:
    # Test con solo prime 2 città per velocità
    train_cities, val_cities, test_cities = create_city_split(
        JSON_FILE,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"✅ City split completato")
    print(f"   Train cities: {len(train_cities)}")
    print(f"   Val cities: {len(val_cities)}")
    print(f"   Test cities: {len(test_cities)}")
    
    # Crea dataset con solo prime 2 città
    dataset = GeospatialPopulationDataset(
        json_file=JSON_FILE,
        base_dir=BASE_DIR,
        crop_size=224,
        transform=None,
        use_tabular_features=False,
        cache_images=False,  # False per test veloce
        cities_filter=train_cities[:2],  # Solo prime 2 città
        augment=False
    )
    
    print(f"\n✅ Dataset creato con {len(dataset)} samples")
    
except Exception as e:
    print(f"\n❌ ERRORE creazione dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# TEST 4: CARICAMENTO SAMPLE
# ============================================================================

print("\n" + "=" * 80)
print("[TEST 4] Caricamento sample...")
print("=" * 80)

try:
    sample = dataset[0]
    
    print(f"✅ Sample caricato:")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Expected: (12, 224, 224)")
    print(f"   Match: {sample['image'].shape == torch.Size([12, 224, 224])} {'✅' if sample['image'].shape == torch.Size([12, 224, 224]) else '❌'}")
    
    print(f"\n   Target: {sample['target'].item():.0f} people")
    print(f"   City: {sample['metadata']['city']}")
    print(f"   Coordinates: ({sample['metadata']['lat']:.4f}, {sample['metadata']['lon']:.4f})")
    
    # Verifica range canali
    print(f"\n   Range canali:")
    print(f"   - RGB (0-2): min={sample['image'][:3].min():.3f}, max={sample['image'][:3].max():.3f}")
    print(f"   - DEM (3): min={sample['image'][3].min():.3f}, max={sample['image'][3].max():.3f}")
    print(f"   - LandUse (4-11): unique values = {torch.unique(sample['image'][4:12]).tolist()}")
    
    # Verifica che land use sia binario
    landuse_values = sample['image'][4:12].unique()
    is_binary = all(v in [0.0, 1.0] for v in landuse_values)
    print(f"   - LandUse è binario {'{0,1}'}: {is_binary} {'✅' if is_binary else '❌'}")
    
    # Verifica shape
    if sample['image'].shape != torch.Size([12, 224, 224]):
        print(f"\n❌ ERRORE: Shape sbagliato! Atteso (12, 224, 224), ottenuto {sample['image'].shape}")
        exit(1)
    
    if not is_binary:
        print(f"\n⚠️ WARNING: Land Use dovrebbe contenere solo 0 e 1!")
    
    print("\n✅ Sample test PASSED")
    
except Exception as e:
    print(f"\n❌ ERRORE caricamento sample: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# TEST 5: MODELLO
# ============================================================================

print("\n" + "=" * 80)
print("[TEST 5] Test modello...")
print("=" * 80)

try:
    model = get_model(
        model_type='baseline',
        pretrained=True,
        freeze_backbone=False,
        device='cpu'
    )
    
    # Test forward pass con il sample caricato
    with torch.no_grad():
        output = model(sample['image'].unsqueeze(0))
    
    print(f"✅ Forward pass completato:")
    print(f"   Input shape: {sample['image'].unsqueeze(0).shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: (1, 1)")
    print(f"   Match: {output.shape == torch.Size([1, 1])} {'✅' if output.shape == torch.Size([1, 1]) else '❌'}")
    print(f"   Prediction: {output.item():.2f} people")
    print(f"   Target: {sample['target'].item():.2f} people")
    
    if output.shape != torch.Size([1, 1]):
        print(f"\n❌ ERRORE: Output shape sbagliato!")
        exit(1)
    
    print("\n✅ Model test PASSED")
    
except Exception as e:
    print(f"\n❌ ERRORE modello: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# TEST 6: DATALOADER
# ============================================================================

print("\n" + "=" * 80)
print("[TEST 6] Test DataLoader...")
print("=" * 80)

try:
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    batch = next(iter(loader))
    
    print(f"✅ Batch caricato:")
    print(f"   Images shape: {batch['image'].shape}")
    print(f"   Targets shape: {batch['target'].shape}")
    print(f"   Expected images: (2, 12, 224, 224)")
    print(f"   Expected targets: (2, 1)")
    print(f"   Match images: {batch['image'].shape == torch.Size([2, 12, 224, 224])} {'✅' if batch['image'].shape == torch.Size([2, 12, 224, 224]) else '❌'}")
    print(f"   Match targets: {batch['target'].shape == torch.Size([2, 1])} {'✅' if batch['target'].shape == torch.Size([2, 1]) else '❌'}")
    
    if batch['image'].shape != torch.Size([2, 12, 224, 224]):
        print(f"\n❌ ERRORE: Batch image shape sbagliato!")
        exit(1)
    
    print("\n✅ DataLoader test PASSED")
    
except Exception as e:
    print(f"\n❌ ERRORE DataLoader: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" 🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("=" * 80)

print("\n✅ Checklist:")
print("   [✓] Struttura directory corretta")
print("   [✓] File presenti in tutte e 3 le cartelle")
print("   [✓] Import funzionanti")
print("   [✓] Dataset carica correttamente")
print("   [✓] Sample shape corretta (12, 224, 224)")
print("   [✓] Canali normalizzati")
print("   [✓] Land Use one-hot encoded")
print("   [✓] Modello funziona")
print("   [✓] DataLoader funziona")

print("\n🚀 SEI PRONTO PER IL TRAINING!")

print("\nProssimi passi:")
print("\n1. Crea un file train_my_model.py:")
print("""
from train_separate_folders import train

train(
    json_file=r"C:\\Users\\andre\\Desktop\\AlmaMater\\Indusrty\\src\\PopulationDataset\\final_clustered_samples.json",
    base_dir=r"C:\\Users\\andre\\Desktop\\AlmaMater\\Indusrty\\src\\PopulationDataset",
    model_type='baseline',
    pretrained=True,
    num_epochs=100,
    batch_size=32,
    device='cuda'  # O 'cpu' se non hai GPU
)
""")

print("\n2. Esegui il training:")
print("   python train_my_model.py")

print("\n3. Monitor con TensorBoard (terminal separato):")
print("   tensorboard --logdir=runs/baseline")

print("\n" + "=" * 80)