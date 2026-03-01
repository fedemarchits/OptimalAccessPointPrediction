"""
run_all.py — Train selected ablation models and generate all visualizations.

For each model this script runs, in order:
  1. Training          (skipped if checkpoint already exists, unless --force-retrain)
  2. Bologna heatmap   (visualize.py  — ground-truth + prediction heatmap)
  3. UMAP analysis     (visualize_umap.py — feature-space coloured by country / population / cluster)

At the end a summary table of all test results is written to outputs/summary/.

Everything you need to download is inside outputs/:
  outputs/{run}/best_model.pth        ← trained weights
  outputs/{run}/test_results.json     ← MAE, RMSE, R²
  outputs/{run}/history.json          ← train/val curves
  outputs/{run}/heatmap/              ← Bologna ground-truth + prediction heatmap
  outputs/{run}/umap/                 ← UMAP plots + metrics.json
  outputs/{run}/logs/                 ← stdout logs for each step
  outputs/summary/                    ← comparison table across all models

Current run set:
  - rgb_only_efficientnet_b3 (ablation: EfficientNet on RGB only — no DEM/LandUse)
  - single_resnet50 / single_efficientnet_b3 / single_convnext_tiny  (re-run with BN)
  - dann_efficientnet_b3    (tuned: max_lambda=0.3, patience=12)
  - dann_convnext_tiny      (DANN on best backbone)
  - crossattn_convnext_tiny (cross-attention on best backbone)
  - multitask_convnext_tiny (new: collaborative cluster-prediction auxiliary task)

Usage:
    python run_all.py \\
        --cache /workspace/PopulationDataset/cache \\
        --json  /workspace/PopulationDataset/final_clustered_samples.json \\
        --base  /workspace/PopulationDataset

    # To re-train models that already have a checkpoint:
    python run_all.py ... --force-retrain
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Experiment registry ───────────────────────────────────────────────────────
#   (run_name, training_script, model_type, backbone)
EXPERIMENTS = [
    # ── DINOv2 (self-supervised ViT-B/14 — primary CV contribution) ──────────
    ("rgb_only_dinov2_vitb14",    "scripts/train_rgb_only_dinov2.py",           "single",    "dinov2_vitb14_rgb"),
    ("single_dinov2_vitb14",      "scripts/train_single_dinov2.py",             "single",    "dinov2_vitb14"),
    ("dann_dinov2_vitb14",        "scripts/train_dann_dinov2.py",               "dann",      "dinov2_vitb14"),

    # ── No image ──────────────────────────────────────────────────────────────
    #("tabular_only",              "scripts/train_tabular_only.py",              "tabular",   "none"),

    # ── Single branch (image only, no tabular) ────────────────────────────────
    ("rgb_only_efficientnet_b3",  "scripts/train_rgb_only_efficientnet.py",     "single",    "efficientnet_b3_rgb"),
    #("single_resnet50",           "scripts/train_single_resnet50.py",           "single",    "resnet50"),
    #("single_efficientnet_b3",    "scripts/train_single_efficientnet.py",       "single",    "efficientnet_b3"),
    ("single_convnext_tiny",      "scripts/train_single_convnext.py",           "single",    "convnext_tiny"),

    # ── Dual branch (image + tabular, simple concat) ──────────────────────────
    #("dual_resnet50",             "scripts/train_dual_resnet50.py",             "dual",      "resnet50"),
    #("dual_efficientnet_b3",      "scripts/train_dual_efficientnet.py",         "dual",      "efficientnet_b3"),
    ("dual_convnext_tiny",        "scripts/train_dual_convnext.py",             "dual",      "convnext_tiny"),

    # ── Dual branch — advanced fusion ─────────────────────────────────────────
    #("film_efficientnet_b3",      "scripts/train_film_efficientnet.py",         "film",      "efficientnet_b3"),
    ("crossattn_efficientnet_b3", "scripts/train_crossattn_efficientnet.py",    "crossattn", "efficientnet_b3"),
    #("crossattn_convnext_tiny",   "scripts/train_crossattn_convnext.py",        "crossattn", "convnext_tiny"),
    #("multitask_convnext_tiny",   "scripts/train_multitask_convnext.py",        "multitask", "convnext_tiny"),

    # ── Domain adaptation ─────────────────────────────────────────────────────
    ("dann_efficientnet_b3",      "scripts/train_dann.py",                      "dann",      "efficientnet_b3"),
    ("dann_convnext_tiny",        "scripts/train_dann_convnext.py",             "dann",      "convnext_tiny"),
]

HEATMAP_CITY = "Bologna, Italy"


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_step(cmd: list[str], log_path: Path, label: str) -> bool:
    """
    Run a subprocess, streaming output to both the console and a log file.
    Returns True on success, False on non-zero exit code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  >>> {label}")
    print(f"      cmd: {' '.join(cmd)}")
    t0 = time.time()

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"  <<< {label}: {status}  ({elapsed:.0f}s)")
    return ok


# ── Summary ───────────────────────────────────────────────────────────────────

def build_summary(output_dir: Path):
    """
    Collect test_results.json from every completed run and write a
    comparison table to outputs/summary/.
    """
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    all_results = {}
    for run_name, _, _, _ in EXPERIMENTS:
        results_path = output_dir / run_name / "test_results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results[run_name] = json.load(f)

    if not all_results:
        print("\nNo completed runs found — summary skipped.")
        return

    with open(summary_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Pretty text table
    header = f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'Loss':>8}"
    sep    = "-" * len(header)
    lines  = [sep, header, sep]
    for run, m in sorted(all_results.items()):
        lines.append(
            f"{run:<30} "
            f"{m.get('mae', float('nan')):>8.1f} "
            f"{m.get('rmse', float('nan')):>8.1f} "
            f"{m.get('r2', float('nan')):>7.4f} "
            f"{m.get('loss', float('nan')):>8.4f}"
        )
    lines.append(sep)
    table = "\n".join(lines)

    with open(summary_dir / "comparison.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(table)
    print(f"{'='*60}")
    print(f"Saved to {summary_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train all ablation models and generate all visualizations."
    )
    parser.add_argument("--cache",         required=True,  help="Path to preprocessed cache dir")
    parser.add_argument("--json",          required=True,  help="Path to final_clustered_samples.json")
    parser.add_argument("--base",          required=True,  help="Base dataset dir (satellite_images/ etc.)")
    parser.add_argument("--output-dir",    default="outputs", help="Root output directory (default: outputs)")
    parser.add_argument("--city",          default=HEATMAP_CITY, help="City for heatmap generation")
    parser.add_argument("--stride",        type=int, default=56, help="Heatmap sliding window stride (default: 56)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train even if checkpoint already exists")
    parser.add_argument("--skip-umap",          action="store_true", help="Skip UMAP visualization")
    parser.add_argument("--skip-heatmap",        action="store_true", help="Skip heatmap generation")
    parser.add_argument("--skip-gradcam",        action="store_true", help="Skip GradCAM visualization")
    parser.add_argument("--skip-shap",           action="store_true", help="Skip tabular SHAP analysis")
    parser.add_argument("--skip-country-errors", action="store_true", help="Skip per-country error analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    py = sys.executable  # use same Python interpreter

    completed, failed, skipped_train = [], [], []

    for run_name, train_script, model_type, backbone in EXPERIMENTS:
        run_dir  = output_dir / run_name
        log_dir  = run_dir / "logs"
        ckpt     = run_dir / "best_model.pth"

        banner = f"\n{'='*60}\n  MODEL: {run_name}\n{'='*60}"
        print(banner)

        # ── 1. Training ──────────────────────────────────────────────────────
        if ckpt.exists() and not args.force_retrain:
            print(f"  Checkpoint found — skipping training ({ckpt})")
            skipped_train.append(run_name)
        else:
            ok = run_step(
                [py, train_script],
                log_dir / "train.log",
                f"Training {run_name}",
            )
            if not ok:
                print(f"  Training FAILED for {run_name} — skipping visualizations.")
                failed.append(run_name)
                continue

        if not ckpt.exists():
            print(f"  No checkpoint at {ckpt} — skipping visualizations.")
            failed.append(run_name)
            continue

        # ── 2. Bologna heatmap ───────────────────────────────────────────────
        if not args.skip_heatmap and model_type != "tabular":
            heatmap_dir = run_dir / "heatmap"
            run_step(
                [
                    py, "visualize.py",
                    "--city",       args.city,
                    "--checkpoint", str(ckpt),
                    "--model",      model_type,
                    "--backbone",   backbone,
                    "--json",       args.json,
                    "--base",       args.base,
                    "--cache",      args.cache,
                    "--out",        str(heatmap_dir),
                    "--stride",     str(args.stride),
                ],
                log_dir / "heatmap.log",
                f"Heatmap for {run_name}",
            )

        # ── 3. UMAP ──────────────────────────────────────────────────────────
        if not args.skip_umap and model_type != "tabular":
            run_step(
                [
                    py, "visualize_umap.py",
                    "--cache",    args.cache,
                    "--run",      run_name,
                    "--model",    model_type,
                    "--backbone", backbone,
                ],
                log_dir / "umap.log",
                f"UMAP for {run_name}",
            )

        # ── 4. GradCAM ───────────────────────────────────────────────────────
        if not args.skip_gradcam and model_type != "tabular":
            run_step(
                [
                    py, "visualize_gradcam.py",
                    "--cache",    args.cache,
                    "--run",      run_name,
                    "--model",    model_type,
                    "--backbone", backbone,
                ],
                log_dir / "gradcam.log",
                f"GradCAM for {run_name}",
            )

        # ── 5. Tabular SHAP (dual-branch models only) ─────────────────────
        if not args.skip_shap and model_type not in ("single", "tabular"):
            shap_cmd = [
                py, "visualize_shap.py",
                "--cache",    args.cache,
                "--run",      run_name,
                "--model",    model_type,
                "--backbone", backbone,
            ]
            if args.json:
                shap_cmd += ["--json", args.json]
            run_step(shap_cmd, log_dir / "shap.log", f"SHAP for {run_name}")

        # ── 6. Per-country error map ─────────────────────────────────────
        if not args.skip_country_errors:
            run_step(
                [
                    py, "visualize_country_errors.py",
                    "--cache",    args.cache,
                    "--run",      run_name,
                    "--model",    model_type,
                    "--backbone", backbone,
                ],
                log_dir / "country_errors.log",
                f"Country errors for {run_name}",
            )

        completed.append(run_name)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    build_summary(output_dir)

    # ── Final status ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Completed ({len(completed)}): {', '.join(completed) or 'none'}")
    print(f"  Failed    ({len(failed)}):    {', '.join(failed) or 'none'}")
    print(f"  Train skipped (ckpt exists): {', '.join(skipped_train) or 'none'}")
    print(f"\n  Download everything in:  {output_dir.resolve()}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
