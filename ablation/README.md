# Population Prediction — Ablation Study

Geospatial population estimation from satellite imagery and OpenStreetMap tabular features.
All experiments use a fixed city-level train/val/test split (seed=42, 70/15/15) so results
are directly comparable. Bologna, Italy is always held out as the test city for heatmaps.

---

## Table of Contents

1. [Task & Data](#task--data)
2. [Model Zoo](#model-zoo)
   - [Single-Branch (image only)](#single-branch-image-only)
   - [RGB-Only Baseline](#rgb-only-baseline)
   - [Dual-Branch Concat Fusion](#dual-branch-concat-fusion)
   - [Cross-Attention Fusion](#cross-attention-fusion)
   - [FiLM Conditioning](#film-conditioning)
   - [DANN — Domain Adversarial Training](#dann--domain-adversarial-training)
3. [Results](#results)
   - [Full Comparison Table](#full-comparison-table)
   - [Modality Ablation](#modality-ablation)
   - [Per-Cluster MAE](#per-cluster-mae)
   - [Discussion](#discussion)
4. [Visualizations](#visualizations)
   - [Bologna Heatmaps](#bologna-heatmaps)
   - [GradCAM](#gradcam)
   - [SHAP / Feature Attribution](#shap--feature-attribution)
   - [UMAP Feature Space](#umap-feature-space)
   - [Per-Country Errors](#per-country-errors)
5. [Setup & Run Guide](#setup--run-guide)

---

## Task & Data

**Goal:** Predict the **residential population reachable within a 15-minute walk** from a given GPS
point in a European city.

**Inputs per sample:**
- A **224×224 px multi-channel image crop** from the city's satellite raster:
  - Channels 0–2: RGB (Sentinel-2, ~10 m/px)
  - Channel 3: DEM elevation (normalised)
  - Channels 4–11: Land-use one-hot (8 classes from CORINE/ESA)
  - **Total: 12 channels**
- **15 OSM tabular features** (road density, building coverage, POI counts, transit
  stops, greenspace ratio, etc.) — used in all dual-branch and attention variants

**Dataset:** 36,546 samples across 80 European cities, city-level split
(train: ~25,600 / val: ~5,600 / test: ~5,300).

**Target:** Population count (log1p-normalised during training; metrics in raw person counts).

**Training config (all models unless noted):**
- AdamW, lr=1e-4, weight_decay=5e-5
- Huber loss δ=1.0
- ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping patience=10 (DANN: 12)
- Batch size 32, up to 100 epochs

---

## Model Zoo

### Single-Branch (image only)

```
image (12ch) → MultiChannelBackbone → (B, feat_dim)
             → RegressionHead (feat_dim→512→256→1) → (B, 1)
```

The backbone's first conv is replaced with a 12-channel equivalent.
Pretrained ImageNet weights are copied to the RGB channels; DEM and land-use
channels are Kaiming-initialised at 10% scale to start as a small perturbation.
A CBAM-style **ChannelAttention** module is applied before the backbone so the
network can learn which of the 12 input channels are most informative.

**Warmup strategy (new):** backbone is frozen for the first 2 epochs so the
regression head can first adapt to ImageNet features. After warm-up, the backbone
is unfrozen with a discriminative learning rate (backbone: 1e-5, head: 1e-4).

Backbone variants tested: ResNet-50, EfficientNet-B3, ConvNeXt-Tiny.

---

### RGB-Only Baseline

```
image (12ch) → x[:, :3]  (discard DEM + land-use)
             → EfficientNet-B3 (standard 3-ch, pretrained weights intact)
             → RegressionHead → (B, 1)
```

Uses the standard pretrained EfficientNet-B3 first conv unchanged — no weight
redistribution. ImageNet normalisation is applied to the RGB channels inside the
backbone so the pretrained weights operate in their expected input range.

**Purpose:** quantify how much the DEM and land-use channels contribute beyond
plain RGB satellite imagery.

---

### Dual-Branch Concat Fusion

```
image   → Backbone → global avg pool → (B, feat_dim)  ──┐
                                                          ├─ concat → FusionHead → (B,1)
tabular → TabularMLP (15→128→64)     → (B, 64)        ──┘
```

The image and tabular branches are independent; their feature vectors are
**concatenated** before a shared regression head. This is the simplest
multi-modal fusion and serves as the baseline for richer variants.

---

### Cross-Attention Fusion

```
image   → Backbone (no pool) → (B, feat_dim, H, W)
        → flatten spatial    → (B, H×W, feat_dim)   ← keys / values
tabular → TabularMLP         → (B, 64) → proj       ← query

Cross-Attention → attended output (B, d_attn)
concat(global_pool, attended, tab_emb) → FusionHead → (B, 1)
```

The OSM neighbourhood context acts as a **query** that attends over the spatial
image feature map. The model learns *where in the image* to focus depending on
local OSM statistics (e.g. high road density → focus on street patterns).

Tested with EfficientNet-B3 (feat_dim=1536, 7×7 map) and ConvNeXt-Tiny
(feat_dim=768, 7×7 map).

---

### FiLM Conditioning

```
image   → Backbone (no pool) → (B, feat_dim, H, W)
tabular → TabularMLP → FiLMGenerator → γ (B, feat_dim), β (B, feat_dim)

modulated = γ * spatial_map + β  → global avg pool → FusionHead → (B, 1)
```

FiLM uses tabular features to generate per-channel **scale and shift** applied
directly to the backbone spatial map before pooling. Rather than choosing *where*
to look, FiLM controls *which feature channels matter* for the neighbourhood profile.
γ initialised to 1, β to 0 (identity start).

---

### DANN — Domain Adversarial Training

```
image + tabular → DualBranchModel backbone + fusion → (B, feat_dim)
                                                    ├→ RegressionHead → population (B,1)
                ← GradientReversalLayer ←           └→ DomainClassifier → country (B, n_countries)
```

A **Gradient Reversal Layer** (GRL) is inserted before a domain classifier that
predicts which country each sample comes from. The GRL negates the gradient during
backprop, so the backbone is simultaneously trained to:
- **Minimise** regression loss (predict population well)
- **Maximise** domain loss (make country features indistinguishable)

This forces the backbone to produce **country-invariant representations**, reducing
the val→test gap caused by city-specific visual shortcuts.

**GRL schedule:** λ ramps from 0 → `dann_max_lambda` over training using
`λ = max_λ · (2/(1 + exp(−10p)) − 1)` where p = epoch/total_epochs.

**Tuned hyperparameters (this run):**
- `dann_max_lambda = 0.3` (reduced from 1.0 — less aggressive gradient reversal)
- `patience = 12` (extended from 10 to allow domain adaptation to stabilise)

Tested with EfficientNet-B3 and ConvNeXt-Tiny backbones.

---

## Results

### Full Comparison Table

All metrics on the **test set** (best val-loss checkpoint).

| Model | Backbone | Tabular | Epochs | MAE ↓ | RMSE ↓ | R² ↑ | Loss ↓ |
|-------|----------|---------|--------|-------|--------|------|--------|
| **DANN** | EfficientNet-B3 | ✓ | 25 | **5,537** | **11,088** | **0.465** | 0.652 |
| DANN | ConvNeXt-Tiny | ✓ | 27 | 6,290 | 11,540 | 0.420 | 0.726 |
| Cross-Attention | ConvNeXt-Tiny | ✓ | 56 | 6,213 | 12,411 | 0.330 | 0.693 |
| Dual-Branch (concat)† | EfficientNet-B3 | ✓ | 43 | 6,077 | 12,915 | 0.379 | 0.643 |
| Cross-Attention† | EfficientNet-B3 | ✓ | 17 | 6,287 | 13,020 | 0.369 | 0.706 |
| FiLM† | EfficientNet-B3 | ✓ | 21 | 7,207 | 14,710 | 0.195 | 0.738 |
| Single-Branch | ConvNeXt-Tiny | ✗ | 30 | 8,796 | 14,009 | 0.146 | 1.565 |
| RGB-Only | EfficientNet-B3 | ✗ | 14 | 8,957 | 15,311 | −0.020 | 1.480 |

† From previous run (outputs/).

---

### Modality Ablation

This table isolates the contribution of each input modality by comparing models
that incrementally add information:

| Model | Extra vs. previous | MAE ↓ | R² ↑ |
|-------|--------------------|-------|------|
| RGB-Only EfficientNet | baseline (3ch RGB) | 8,957 | −0.02 |
| Single ConvNeXt (12ch) | + DEM + land-use | 8,796 | +0.17 |
| Dual-Branch / CrossAttn | + OSM tabular | ~6,200 | ~0.35–0.42 |
| DANN EfficientNet | + domain invariance | **5,537** | **0.465** |

**Key finding:** OSM tabular features are the primary driver of performance,
reducing MAE by ~30% over any image-only model. DEM and land-use contribute only
marginally on top of RGB. Domain adaptation (DANN) provides a further meaningful
improvement by removing country-specific biases.

---

### Per-Cluster MAE (test set)

Clusters represent city typologies derived from OSM: 0 = dense urban core,
4 = suburban/peri-urban.

| Model | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|-------|-----------|-----------|-----------|-----------|-----------|
| DANN EfficientNet | **4,482** | **6,248** | **6,165** | **6,229** | **4,506** |
| DANN ConvNeXt | 5,007 | 7,412 | 7,136 | 6,712 | 5,104 |
| CrossAttn ConvNeXt | 4,818 | 7,080 | 7,109 | 6,854 | 5,132 |
| Single ConvNeXt | 7,744 | 10,454 | 11,035 | 8,596 | 6,024 |
| RGB-Only | 6,840 | 9,994 | 11,081 | 9,924 | 6,843 |

DANN EfficientNet is the best model across every cluster. The largest improvements
over single-branch are in Clusters 1–3 (mixed urban / industrial areas) where
visual patterns alone are most ambiguous.

---

### Discussion

**Why DANN EfficientNet is the best model:**
DANN with the tuned `max_lambda=0.3` finds a better balance between regression
accuracy and domain invariance compared to the previous run (λ=1.0 was too aggressive,
causing the domain classifier to collapse to near-random accuracy). At λ=0.3, the
backbone learns representations that are both predictive and country-invariant,
which generalises better to the test set.

**Why DANN EfficientNet beats DANN ConvNeXt:**
Even though ConvNeXt-Tiny has a stronger architecture for image tasks, EfficientNet-B3
with DANN training generalises better across countries. This may be because
EfficientNet-B3's wider feature dimension (1536 vs 768) provides more capacity
for the domain classifier to identify country-specific signals, making the adversarial
training more effective.

**Why single-branch models struggle:**
Image-only models memorise city-specific visual textures and fail to generalise
to validation/test cities (val loss stays flat throughout training). OSM tabular
features provide city-agnostic signals (building density, road hierarchy, amenity
counts) that transfer across cities regardless of visual appearance.

**RGB-Only vs. 12-channel single-branch:**
The marginal improvement from adding DEM and land-use (R² −0.02 → 0.15) confirms
these modalities provide limited discriminative signal on their own — their value
is primarily to help the image backbone learn better representations when combined
with the tabular branch.

**Cross-Attention ConvNeXt:**
Despite training for 56 epochs (vs. 25–27 for DANN), CrossAttn ConvNeXt achieves
lower R² (0.330) than both DANN models. The attention mechanism adds complexity
but the spatial feature maps at ConvNeXt's 7×7 resolution may not provide enough
spatial diversity for the tabular query to meaningfully attend to.

---

## Visualizations

> All outputs are in `../outputs2/{run_name}/`.

---

### Bologna Heatmaps

Dense sliding-window inference over the full Bologna satellite image.
Left: ground-truth sample locations coloured by actual population.
Right: model predictions as a continuous heatmap.

#### DANN EfficientNet-B3 (best model)

**Ground truth**
![Ground truth](../outputs2/dann_efficientnet_b3/heatmap/Bologna_Italy_groundtruth.png)

**Prediction heatmap**
![DANN EfficientNet heatmap](../outputs2/dann_efficientnet_b3/heatmap/Bologna_Italy_heatmap.png)

#### DANN ConvNeXt-Tiny

![DANN ConvNeXt heatmap](../outputs2/dann_convnext_tiny/heatmap/Bologna_Italy_heatmap.png)

#### Cross-Attention ConvNeXt-Tiny

![CrossAttn ConvNeXt heatmap](../outputs2/crossattn_convnext_tiny/heatmap/Bologna_Italy_heatmap.png)

#### Single-Branch ConvNeXt-Tiny (image only)

![Single ConvNeXt heatmap](../outputs2/single_convnext_tiny/heatmap/Bologna_Italy_heatmap.png)

#### RGB-Only EfficientNet-B3

![RGB-only heatmap](../outputs2/rgb_only_efficientnet_b3/heatmap/Bologna_Italy_heatmap.png)

---

### GradCAM

GradCAM backpropagates the population prediction through the backbone's last
spatial layer to highlight which image regions drive each prediction.

#### DANN EfficientNet — Best predictions (lowest absolute error)

![DANN EfficientNet best](../outputs2/dann_efficientnet_b3/gradcam/best_predictions.png)

#### DANN EfficientNet — Worst predictions

![DANN EfficientNet worst](../outputs2/dann_efficientnet_b3/gradcam/worst_predictions.png)

#### DANN EfficientNet — Average CAM: high vs. low population density

![DANN EfficientNet avg CAM](../outputs2/dann_efficientnet_b3/gradcam/average_high_vs_low.png)

#### Cross-Attention ConvNeXt — Best predictions

![CrossAttn best](../outputs2/crossattn_convnext_tiny/gradcam/best_predictions.png)

---

### SHAP / Feature Attribution

Gradient × Input attribution over the full test set, measuring how much each
OSM tabular feature contributes to each prediction.

#### DANN EfficientNet — Feature importance (mean |attribution|)

![DANN EfficientNet feature importance](../outputs2/dann_efficientnet_b3/shap/feature_importance.png)

#### DANN EfficientNet — Attribution direction (signed mean)

Red = pushes prediction up, blue = pushes prediction down.

![DANN EfficientNet attribution direction](../outputs2/dann_efficientnet_b3/shap/attribution_direction.png)

#### DANN EfficientNet — Beeswarm (per-sample distribution)

![DANN EfficientNet beeswarm](../outputs2/dann_efficientnet_b3/shap/beeswarm.png)

#### DANN EfficientNet — Per-cluster attribution heatmap

![DANN EfficientNet cluster heatmap](../outputs2/dann_efficientnet_b3/shap/cluster_heatmap.png)

#### Cross-Attention ConvNeXt — Feature importance

![CrossAttn feature importance](../outputs2/crossattn_convnext_tiny/shap/feature_importance.png)

---

### UMAP Feature Space

UMAP of backbone features extracted on the test set, coloured by country,
population quartile, and urban cluster. Measures whether the model's learned
representation separates by domain (country) or by the target signal (population).

#### DANN EfficientNet — by country

A well-calibrated DANN model should show *mixed* country colours (no country
clusters), indicating domain-invariant features.

![DANN EfficientNet UMAP country](../outputs2/dann_efficientnet_b3/umap/umap_by_country.png)

#### DANN EfficientNet — by population

![DANN EfficientNet UMAP population](../outputs2/dann_efficientnet_b3/umap/umap_by_population.png)

#### DANN EfficientNet — by cluster

![DANN EfficientNet UMAP cluster](../outputs2/dann_efficientnet_b3/umap/umap_by_cluster.png)

#### Single-Branch ConvNeXt — by country (for comparison)

Without domain adaptation, country clusters should be more visible.

![Single ConvNeXt UMAP country](../outputs2/single_convnext_tiny/umap/umap_by_country.png)

---

### Per-Country Errors

MAE broken down by country on the test set.

#### DANN EfficientNet

![DANN EfficientNet country errors](../outputs2/dann_efficientnet_b3/country_errors/country_mae_bar.png)

#### DANN ConvNeXt

![DANN ConvNeXt country errors](../outputs2/dann_convnext_tiny/country_errors/country_mae_bar.png)

#### Cross-Attention ConvNeXt

![CrossAttn country errors](../outputs2/crossattn_convnext_tiny/country_errors/country_mae_bar.png)

---

## Setup & Run Guide

### 0. Upload code to Vast.ai

```bash
rsync -avz -e "ssh -p PORT" \
    /path/to/OptimalAccessPointPrediction/ablation/ \
    root@IP:/workspace/OptimalAccessPointPrediction/ablation/
```

### 1. Install dependencies

```bash
cd /workspace/OptimalAccessPointPrediction/ablation
bash setup_vastai.sh
pip install matplotlib scipy umap-learn scikit-learn geopandas timm
```

### 2. Download the dataset

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='fedemarchits/populationAccess',
    repo_type='dataset',
    local_dir='/workspace/PopulationDataset'
)
"
# then unzip if needed
unzip /workspace/PopulationDataset/PopulationDataset.zip -d /workspace/PopulationDataset/
```

### 3. Preprocess the dataset (one-time, ~20–40 min)

```bash
python data/preprocess.py \
  --json /workspace/PopulationDataset/final_clustered_samples.json \
  --base /workspace/PopulationDataset \
  --out  /workspace/PopulationDataset/cache
```

### 4. Log in to Weights & Biases

```bash
wandb login
# or: export WANDB_MODE=disabled
```

### 5. Run all experiments

```bash
python run_all.py \
  --cache /workspace/PopulationDataset/cache \
  --json  /workspace/PopulationDataset/final_clustered_samples.json \
  --base  /workspace/PopulationDataset
```

The `EXPERIMENTS` list in `run_all.py` controls which models are trained.
Models with existing checkpoints are skipped automatically unless `--force-retrain` is passed.

To skip specific visualization steps:
```bash
python run_all.py ... --skip-umap --skip-gradcam --skip-shap --skip-country-errors
```

### 6. Copy results back locally

```bash
rsync -avz -e "ssh -p PORT" \
  root@IP:/workspace/OptimalAccessPointPrediction/ablation/outputs/ \
  ./outputs/
```
