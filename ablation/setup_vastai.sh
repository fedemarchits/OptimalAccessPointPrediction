#!/usr/bin/env bash
# =============================================================================
#  Vast.ai setup script — run this ONCE on the rented instance
#  Assumes CUDA/PyTorch base image (torch is likely pre-installed)
# =============================================================================
set -e

echo "===== 1. Upgrading pip ====="
pip install --upgrade pip

echo "===== 2. Installing Python dependencies ====="
pip install \
    "torch>=2.0" \
    "torchvision>=0.15" \
    "timm>=0.9" \
    "rasterio>=1.3" \
    "numpy>=1.24" \
    "tqdm>=4.65" \
    "tensorboard>=2.13" \
    "wandb>=0.15"

echo "===== 3. Verifying dataset structure ====="
DATASET=/workspace/PopulationDataset

missing=0
for d in satellite_images dem_height segmentation_land_use; do
    if [ -d "$DATASET/$d" ]; then
        count=$(ls "$DATASET/$d" | wc -l)
        echo "  ✓ $d  ($count files)"
    else
        echo "  ✗ MISSING: $DATASET/$d"
        missing=1
    fi
done

if [ -f "$DATASET/final_clustered_samples.json" ]; then
    python3 -c "
import json
d = json.load(open('$DATASET/final_clustered_samples.json'))
n = len(d)
n_osm = sum(1 for s in d if 'osm_features' in s and s['osm_features'])
print(f'  ✓ JSON: {n} samples, {n_osm} with osm_features')
if n_osm == 0:
    print('  ⚠  WARNING: no osm_features found — dual-branch will use zero features')
"
else
    echo "  ✗ MISSING: $DATASET/final_clustered_samples.json"
    missing=1
fi

if [ $missing -eq 1 ]; then
    echo ""
    echo "  Some files are missing. See the upload instructions below."
    exit 1
fi

echo ""
echo "===== 4. Logging into W&B (optional) ====="
echo "  Run:  wandb login"
echo "  Or set WANDB_API_KEY env var."
echo ""
echo "===== Setup complete! ====="
echo ""
echo "To run an experiment:"
echo "  cd /workspace/ablation"
echo "  python scripts/train_dual_efficientnet.py"
