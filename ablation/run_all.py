"""
run_all.py — Train selected ablation models and evaluate on the test set.

For each model this script runs, in order:
  1. Training   (skipped if checkpoint already exists, unless --force-retrain)

At the end a summary table of all test results is written to outputs/summary/.

Outputs per run:
  outputs/{run}/best_model.pth     ← trained weights
  outputs/{run}/test_results.json  ← MAE, RMSE, R²
  outputs/{run}/history.json       ← train/val loss curves
  outputs/{run}/logs/train.log     ← full stdout log
  outputs/summary/                 ← comparison table across all models

Usage:
    python run_all.py \\
        --cache /workspace/PopulationDataset/cache \\
        --json  /workspace/PopulationDataset/final_clustered_samples.json \\
        --base  /workspace/PopulationDataset

    # Re-train even if a checkpoint already exists:
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
# EXPERIMENTS = [
#     # ── DINOv2 (self-supervised ViT-B/14 — primary CV contribution) ──────────
#     # ("rgb_only_dinov2_vitb14",    "scripts/train_rgb_only_dinov2.py",           "single",    "dinov2_vitb14_rgb"),
#     # ("single_dinov2_vitb14",      "scripts/train_single_dinov2.py",             "single",    "dinov2_vitb14"),
#     ("dual_dinov2_vitb14",      "scripts/train_dual_dinov2.py",             "dual",    "dinov2_vitb14"),
#     # ("dann_dinov2_vitb14",        "scripts/train_dann_dinov2.py",               "dann",      "dinov2_vitb14"),

#     # ── No image ──────────────────────────────────────────────────────────────
#     # ("tabular_only",              "scripts/train_tabular_only.py",              "tabular",   "none"),

#     # ── Single branch (image only, no tabular) ────────────────────────────────
#     # ("rgb_only_efficientnet_b3",  "scripts/train_rgb_only_efficientnet.py",     "single",    "efficientnet_b3_rgb"),
#     ("single_resnet50",           "scripts/train_single_resnet50.py",           "single",    "resnet50"),
#     ("single_efficientnet_b3",    "scripts/train_single_efficientnet.py",       "single",    "efficientnet_b3"),
#     # ("single_convnext_tiny",      "scripts/train_single_convnext.py",           "single",    "convnext_tiny"),

#     # ── Dual branch (image + tabular, simple concat) ──────────────────────────
#     ("dual_resnet50",             "scripts/train_dual_resnet50.py",             "dual",      "resnet50"),
#     ("dual_efficientnet_b3",      "scripts/train_dual_efficientnet.py",         "dual",      "efficientnet_b3"),
#     # ("dual_convnext_tiny",        "scripts/train_dual_convnext.py",             "dual",      "convnext_tiny"),

#     # ── Dual branch — advanced fusion ─────────────────────────────────────────
#     ("film_efficientnet_b3",      "scripts/train_film_efficientnet.py",         "film",      "efficientnet_b3"),
#     # ("crossattn_efficientnet_b3", "scripts/train_crossattn_efficientnet.py",    "crossattn", "efficientnet_b3"),
#     ("crossattn_convnext_tiny",   "scripts/train_crossattn_convnext.py",        "crossattn", "convnext_tiny"),

#     # ── Domain adaptation ─────────────────────────────────────────────────────
#     ("dann_efficientnet_b3",      "scripts/train_dann.py",                      "dann",      "efficientnet_b3"),
#     ("dann_convnext_tiny",        "scripts/train_dann_convnext.py",             "dann",      "convnext_tiny"),
# ]

EXPERIMENTS = [
    
    ("single_resnet50",           "scripts/train_single_resnet50.py",           "single",    "resnet50"),
    ("single_efficientnet_b3",    "scripts/train_single_efficientnet.py",       "single",    "efficientnet_b3"),
    ("dual_dinov2_vitb14",      "scripts/train_dual_dinov2.py",             "dual",    "dinov2_vitb14"),
    
    # ("dual_resnet50",             "scripts/train_dual_resnet50.py",             "dual",      "resnet50"),
    # ("dual_efficientnet_b3",      "scripts/train_dual_efficientnet.py",         "dual",      "efficientnet_b3"),
    
    # ("film_efficientnet_b3",      "scripts/train_film_efficientnet.py",         "film",      "efficientnet_b3"),
    
    # ("crossattn_convnext_tiny",   "scripts/train_crossattn_convnext.py",        "crossattn", "convnext_tiny"),

    # ("dann_efficientnet_b3",      "scripts/train_dann.py",                      "dann",      "efficientnet_b3"),
    # ("dann_convnext_tiny",        "scripts/train_dann_convnext.py",             "dann",      "convnext_tiny"),
]


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
        description="Train all ablation models and evaluate on the test set."
    )
    parser.add_argument("--cache",         required=True,  help="Path to preprocessed cache dir")
    parser.add_argument("--json",          required=True,  help="Path to final_clustered_samples.json")
    parser.add_argument("--base",          required=True,  help="Base dataset dir")
    parser.add_argument("--output-dir",    default="outputs", help="Root output directory (default: outputs)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train even if checkpoint already exists")
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
                print(f"  Training FAILED for {run_name} — skipping evaluation.")
                failed.append(run_name)
                continue

        if not ckpt.exists():
            print(f"  No checkpoint at {ckpt} — skipping evaluation.")
            failed.append(run_name)
            continue

        completed.append(run_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    build_summary(output_dir)

    # ── Final status ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Completed ({len(completed)}): {', '.join(completed) or 'none'}")
    print(f"  Failed    ({len(failed)}):    {', '.join(failed) or 'none'}")
    print(f"  Train skipped (ckpt exists): {', '.join(skipped_train) or 'none'}")
    print(f"\n  Outputs in:  {output_dir.resolve()}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
