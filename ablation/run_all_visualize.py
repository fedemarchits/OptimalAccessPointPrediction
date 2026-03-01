"""
Run visualize.py for every model in outputs3, for two cities:
  - Bologna, Italy  (training city — familiar reference)
  - Milan, Italy    (test city — fair evaluation)

Cross-attention models produce a 5-patch attention figure.
tabular_only is skipped (no image branch).

Usage:
    cd ablation/
    python run_all_visualize.py
"""

import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR.parent / "outputs3"
JSON        = str(BASE_DIR.parent / "data" / "final_clustered_samples.json")
IMG_BASE    = str(BASE_DIR.parent / "imgs")
CACHE       = str(BASE_DIR.parent / "preprocessed_cache")
OSM_DIR     = str(BASE_DIR.parent / "data" / "osm_features")

CITIES = [
    "Bologna, Italy",   # training city — familiar reference
    "Milan, Italy",     # test city — fair evaluation
]

# run → (model_type, backbone)
MODELS = {
    "crossattn_efficientnet_b3": ("crossattn", "efficientnet_b3"),
    "dann_dinov2_vitb14":        ("dann",      "dinov2_vitb14"),
    "dual_convnext_tiny":        ("dual",      "convnext_tiny"),
    "rgb_only_dinov2_vitb14":    ("single",    "dinov2_vitb14_rgb"),
    "rgb_only_efficientnet_b3":  ("single",    "efficientnet_b3_rgb"),
    "single_convnext_tiny":      ("single",    "convnext_tiny"),
    "single_dinov2_vitb14":      ("single",    "dinov2_vitb14"),
    "single_efficientnet_b3":    ("single",    "efficientnet_b3"),
    # tabular_only skipped — no image branch
}

STRIDE = "56"

# ── Runner ────────────────────────────────────────────────────────────────────

def run(run_name, model_type, backbone, city):
    ckpt    = str(OUTPUTS_DIR / run_name / "best_model.pth")
    out_dir = str(OUTPUTS_DIR / run_name / "visualizations")
    city_slug = city.replace(", ", "_").replace(" ", "_")

    print(f"\n{'='*65}")
    print(f"  {run_name}  |  {city}")
    print(f"{'='*65}")

    cmd = [
        sys.executable, "visualize.py",
        "--city",       city,
        "--checkpoint", ckpt,
        "--model",      model_type,
        "--backbone",   backbone,
        "--json",       JSON,
        "--base",       IMG_BASE,
        "--cache",      CACHE,
        "--osm-dir",    OSM_DIR,
        "--stride",     STRIDE,
        "--out",        out_dir,
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"  !! FAILED: {run_name} / {city}")


if __name__ == "__main__":
    total = len(MODELS) * len(CITIES)
    done  = 0
    for run_name, (model_type, backbone) in MODELS.items():
        for city in CITIES:
            run(run_name, model_type, backbone, city)
            done += 1
            print(f"\n[{done}/{total} done]")

    print(f"\n{'='*65}")
    print("All visualizations complete.")
    print(f"Outputs in: {OUTPUTS_DIR}/<model>/visualizations/")
