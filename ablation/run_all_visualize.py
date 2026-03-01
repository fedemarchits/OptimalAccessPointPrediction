"""
Run visualize.py for every model in outputs3, for two cities:
  - Bologna, Italy  (training city — familiar reference)
  - Milan, Italy    (test city — fair evaluation)

Cross-attention models produce 5 separate attention-patch images.
tabular_only is skipped (no image branch).

Usage:
    cd ablation/

    # Full run (heatmaps + attention)
    python run_all_visualize.py

    # Attention maps only — much faster, skips sliding-window heatmap
    python run_all_visualize.py --attention-only

    # Only cross-attention model
    python run_all_visualize.py --attention-only --model crossattn_efficientnet_b3
"""

import argparse
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

def run(run_name, model_type, backbone, city, no_heatmap=False):
    ckpt    = str(OUTPUTS_DIR / run_name / "best_model.pth")
    out_dir = str(OUTPUTS_DIR / run_name / "visualizations")

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
    if no_heatmap:
        cmd.append("--no-heatmap")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"  !! FAILED: {run_name} / {city}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", action="store_true",
                        help="Skip heatmaps — only regenerate attention maps (fast)")
    parser.add_argument("--model", default=None,
                        help="Run only this model key (e.g. crossattn_efficientnet_b3)")
    args = parser.parse_args()

    models = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model '{args.model}'. Available: {list(MODELS)}")
            sys.exit(1)
        models = {args.model: MODELS[args.model]}

    total = len(models) * len(CITIES)
    done  = 0
    for run_name, (model_type, backbone) in models.items():
        for city in CITIES:
            run(run_name, model_type, backbone, city, no_heatmap=args.attention_only)
            done += 1
            print(f"\n[{done}/{total} done]")

    print(f"\n{'='*65}")
    print("All visualizations complete.")
    print(f"Outputs in: {OUTPUTS_DIR}/<model>/visualizations/")
