"""
GradCAM visualization for population prediction models.

Backpropagates the scalar population prediction through the backbone's
final spatial feature layer to highlight which image regions drive each
prediction.  Produces three outputs per run:

  gradcam/best_predictions.png   — samples with lowest absolute error
  gradcam/worst_predictions.png  — samples with highest absolute error
  gradcam/random_samples.png     — random sample
  gradcam/average_high_vs_low.png — avg CAM for top-25% vs bottom-25% pop

Works with all model types and backbones (ConvNeXt, EfficientNet, ResNet50).

Usage:
    python visualize_gradcam.py \\
        --cache   /workspace/PopulationDataset/cache \\
        --run     dual_convnext_tiny \\
        --model   dual \\
        --backbone convnext_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from tqdm import tqdm


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",       required=True)
    p.add_argument("--run",         required=True)
    p.add_argument("--model",       required=True,
                   choices=["single", "dual", "crossattn", "film", "dann", "multitask"])
    p.add_argument("--backbone",    default="convnext_tiny")
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--checkpoint",  default=None,
                   help="Path to best_model.pth (overrides default outputs/<run>/best_model.pth)")
    p.add_argument("--n-samples",   type=int, default=15,
                   help="Total samples to show (split equally: best/worst/random)")
    p.add_argument("--n-avg",       type=int, default=40,
                   help="Samples for average high-vs-low CAM")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type, backbone, ckpt_path, device):
    from configs import (SingleBranchConfig, DualBranchConfig, CrossAttnConfig,
                         FiLMConfig, DANNConfig, MultiTaskConfig)
    from models import build_model

    cfg_map = {
        "single":    SingleBranchConfig(backbone=backbone),
        "dual":      DualBranchConfig(backbone=backbone),
        "crossattn": CrossAttnConfig(backbone=backbone),
        "film":      FiLMConfig(backbone=backbone),
        "dann":      DANNConfig(backbone=backbone),
        "multitask": MultiTaskConfig(backbone=backbone),
    }
    model = build_model(cfg_map[model_type], device=device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── GradCAM target layer ──────────────────────────────────────────────────────

def _gradcam_target(model):
    """
    Select the layer whose output is used as the spatial feature map.

    - ConvNeXt / EfficientNet: backbone.backbone  (timm, global_pool="" → 4D)
    - ResNet50:                 backbone.features[7]  (layer4 → (B, 2048, 7, 7))
    - DINOv2 (ViT):            backbone.backbone.blocks[-1]
                                (last transformer block → (B, 1+256, 768))
    """
    bb = model.backbone
    name = type(bb).__name__
    if "ResNet" in name:
        return bb.features[7]           # layer4
    elif "DINOv2" in name:
        return bb.backbone.blocks[-1]   # last ViT block → (B, N, D)
    else:
        return bb.backbone              # timm CNN with global_pool="" → 4D


# ── ViT gradient saliency (DINOv2) ────────────────────────────────────────────

def _vit_gradient_saliency(model, image, tabular, model_type, device):
    """
    For DINOv2/ViT backbones: gradient of the final prediction w.r.t. the
    input image pixels.

    Unlike CLS attention (which reflects DINOv2's pretrained priors on natural
    photos), this directly measures which pixels causally affect the walkability
    prediction through the full finetuned model, including the fusion head.

    Returns:
        cam   np.ndarray (224, 224) in [0, 1]
        pred  float — raw (log-space) prediction
    """
    model.eval()
    img = image.unsqueeze(0).to(device).requires_grad_(True)
    tab = tabular.unsqueeze(0).to(device)

    model.zero_grad()
    if model_type == "single":
        pred = model(img)
    elif model_type == "dann":
        pred, _ = model(img, tab, 0.0)
    else:
        pred = model(img, tab)

    pred.squeeze().backward()

    # Max over channels → (H, W) saliency map
    sal = img.grad[0].abs().max(dim=0)[0]   # (H, W) or (224, 224) already
    sal = sal.detach().cpu().float()

    # Smooth slightly to reduce pixel-level noise
    sal = F.avg_pool2d(sal.unsqueeze(0).unsqueeze(0),
                       kernel_size=11, stride=1, padding=5).squeeze()

    sal = sal.numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    # Resize to 224×224 in case crop was different
    sal = F.interpolate(
        torch.tensor(sal).unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    return sal, float(pred.squeeze().item())


# ── Core GradCAM ──────────────────────────────────────────────────────────────

def gradcam(model, image, tabular, model_type, device):
    """
    Compute GradCAM for a single (1-sample) batch.
    For DINOv2/ViT backbones, uses CLS attention map instead (GradCAM
    produces uniform maps for transformers due to gradient averaging).

    Returns:
        cam   np.ndarray (224, 224) in [0, 1]
        pred  float — raw (log-space) prediction
    """
    # ViT backbones: use gradient saliency instead of GradCAM.
    # CLS attention reflects DINOv2's pretrained priors (not finetuned for
    # walkability), so it localises to visually salient regions unrelated to
    # the prediction.  Gradient saliency propagates through the full model.
    if "DINOv2" in type(model.backbone).__name__:
        return _vit_gradient_saliency(model, image, tabular, model_type, device)

    model.eval()
    image   = image.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)

    acts_buf  = [None]
    grads_buf = [None]
    target    = _gradcam_target(model)

    def _fwd(mod, inp, out):
        acts_buf[0] = out

    def _bwd(mod, gin, gout):
        grads_buf[0] = gout[0]

    h1 = target.register_forward_hook(_fwd)
    h2 = target.register_full_backward_hook(_bwd)

    try:
        model.zero_grad()

        if model_type == "single":
            pred = model(image)
        elif model_type == "dann":
            pred, _ = model(image, tabular, 0.0)
        else:
            pred = model(image, tabular)

        pred = pred.squeeze()
        pred.backward()

        acts  = acts_buf[0]   # (1, C, H, W)
        grads = grads_buf[0]  # (1, C, H, W)

        # GAP the gradients → channel weights
        w   = grads.mean(dim=[2, 3], keepdim=True)         # (1, C, 1, 1)
        cam = F.relu((w * acts).sum(1).squeeze(0))          # (H, W)

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()

    finally:
        h1.remove()
        h2.remove()

    return cam, float(pred.item())


# ── Visualization helpers ─────────────────────────────────────────────────────

def _overlay(rgb, cam, alpha=0.45):
    heatmap = mpl_cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heatmap, 0, 1)


def _plot_grid(samples, out_path, title, cam_label="Attention Map"):
    """
    samples: list of dicts {rgb, cam, pred, target, country, cluster}
    Columns: original | cam_label | overlay
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, d in enumerate(samples):
        rgb, cam = d["rgb"], d["cam"]
        err = abs(d["pred"] - d["target"])
        err_pct = 100 * err / max(d["target"], 1)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"GT = {d['target']:.0f}  ({d['country']}, cl {d['cluster']})",
                              fontsize=9)
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title(cam_label, fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        axes[i, 2].imshow(_overlay(rgb, cam))
        axes[i, 2].set_title(f"Pred = {d['pred']:.0f}  |  Err = {err:.0f} ({err_pct:.0f}%)",
                              fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def _plot_avg(high_cams, low_cams, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cams, label in [
        (axes[0], high_cams, f"High population (top 25%,  n={len(high_cams)})"),
        (axes[1], low_cams,  f"Low population  (bottom 25%, n={len(low_cams)})"),
    ]:
        if cams:
            avg = np.mean(np.stack(cams), axis=0)
            im  = ax.imshow(avg, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    plt.suptitle("Average GradCAM — High vs Low Population Density",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path("outputs") / args.run / "gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}  |  Run: {args.run}")

    ckpt = Path(args.checkpoint) if args.checkpoint else Path("outputs") / args.run / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = load_model(args.model, args.backbone, ckpt, device)

    from data.dataset import CachedDataset
    uses_tabular = args.model != "single"
    test_ds = CachedDataset(args.cache, "test", use_tabular=uses_tabular)
    denorm  = test_ds.denormalize_target

    # ── Fast pass: collect all predictions (no GradCAM yet) ──────────────
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    all_preds, all_targets = [], []
    print("Running inference for sample selection...")
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            imgs = batch["image"].to(device)
            tab  = batch["tabular"].to(device)
            tgts = batch["target"].squeeze()

            if args.model == "single":
                p = model(imgs)
            elif args.model == "dann":
                p, _ = model(imgs, tab, 0.0)
            else:
                p = model(imgs, tab)

            all_preds.extend(denorm(p.squeeze().cpu()).tolist())
            all_targets.extend(denorm(tgts.cpu()).tolist())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    errors  = np.abs(preds - targets)
    rng     = np.random.default_rng(42)

    k = max(1, args.n_samples // 3)
    sel_best   = np.argsort(errors)[:k]
    sel_worst  = np.argsort(errors)[-k:][::-1]
    sel_random = rng.choice(len(preds), k, replace=False)

    def _sample(idx):
        item = test_ds[idx]
        cam, pred_raw = gradcam(model, item["image"], item["tabular"],
                                args.model, device)
        return {
            "rgb":     np.clip(item["image"][:3].numpy().transpose(1, 2, 0), 0, 1),
            "cam":     cam,
            "pred":    float(denorm(torch.tensor(pred_raw))),
            "target":  float(denorm(item["target"].squeeze())),
            "country": item["metadata"]["country"],
            "cluster": int(item["metadata"]["cluster"]),
        }

    is_vit    = "DINOv2" in type(model.backbone).__name__
    cam_label = "Gradient Saliency" if is_vit else "GradCAM"
    tag       = "Saliency" if is_vit else "GradCAM"

    print(f"{tag} for {k} best predictions...")
    _plot_grid([_sample(i) for i in tqdm(sel_best, leave=False)],
               out_dir / "best_predictions.png",
               f"{tag} — Best Predictions  [{args.run}]",
               cam_label=cam_label)

    print(f"{tag} for {k} worst predictions...")
    _plot_grid([_sample(i) for i in tqdm(sel_worst, leave=False)],
               out_dir / "worst_predictions.png",
               f"{tag} — Worst Predictions  [{args.run}]",
               cam_label=cam_label)

    print(f"{tag} for {k} random samples...")
    _plot_grid([_sample(i) for i in tqdm(sel_random, leave=False)],
               out_dir / "random_samples.png",
               f"{tag} — Random Samples  [{args.run}]",
               cam_label=cam_label)

    # ── Average CAM: high vs low population ──────────────────────────────
    q75, q25 = np.percentile(targets, 75), np.percentile(targets, 25)
    hi_pool  = np.where(targets >= q75)[0]
    lo_pool  = np.where(targets <= q25)[0]
    n_avg    = min(args.n_avg, len(hi_pool), len(lo_pool))
    hi_sel   = rng.choice(hi_pool, n_avg, replace=False)
    lo_sel   = rng.choice(lo_pool, n_avg, replace=False)

    print(f"Average {tag} (n={n_avg} each)...")
    hi_cams = [gradcam(model, test_ds[i]["image"], test_ds[i]["tabular"],
                       args.model, device)[0]
               for i in tqdm(hi_sel, desc="High-pop", leave=False)]
    lo_cams = [gradcam(model, test_ds[i]["image"], test_ds[i]["tabular"],
                       args.model, device)[0]
               for i in tqdm(lo_sel, desc="Low-pop",  leave=False)]
    _plot_avg(hi_cams, lo_cams, out_dir / "average_high_vs_low.png")

    print(f"\n{tag} done → {out_dir}/")


if __name__ == "__main__":
    main()
