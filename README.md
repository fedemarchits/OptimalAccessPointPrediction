# Walkability Population Prediction from Satellite Imagery and OSM Features

**Predicting the population reachable within a 15-minute walk** from any urban point, using a fusion of multi-channel satellite imagery (Sentinel-2 + DEM + land-use) and OpenStreetMap tabular features, across 80 European cities.

> Inspired by: Doda et al., _"Interpretable deep learning for consistent large-scale urban population estimation using Earth observation data"_, Int. Journal of Applied Earth Observation and Geoinformation 128 (2024) 103731. [[DOI]](https://doi.org/10.1016/j.jag.2024.103731)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Construction](#2-dataset-construction)
3. [Data Preprocessing & Cache](#3-data-preprocessing--cache)
4. [Model Architecture](#4-model-architecture)
5. [Training Setup](#5-training-setup)
6. [Results](#6-results)
7. [Visualizations](#7-visualizations)
8. [Reproduction](#8-reproduction)
9. [References](#9-references)

---

## 1. Problem Statement

Urban walkability — the ability to reach destinations on foot — is a key indicator of urban quality of life and sustainable mobility. A granular, census-free estimate of _how many people are reachable within a 15-minute walk_ from any given point enables urban planners, transport engineers, and policymakers to evaluate pedestrian accessibility at city scale without relying on costly survey data.

This project builds a regression model that predicts this walkability score directly from:

- **Satellite imagery** (Sentinel-2 RGB + DEM elevation + ESA land-use), forming a 12-channel spatial tensor
- **OSM tabular features** (road network, buildings, POIs, transport) extracted within a 1120 m radius buffer

The task is challenging because the dominant signal comes from the pedestrian network topology (captured by OSM), while the image contributes complementary context about built density and land cover. A city-level train/val/test split creates a genuine domain-shift evaluation: models must generalise to **unseen cities**, not unseen neighbourhoods.

---

## 2. Dataset Construction

### 2.1 City Selection

80 European cities were selected to ensure geographic and morphological diversity — covering Northern, Southern, Eastern and Western Europe, with cities ranging from compact historic centres (Bologna, Bruges) to large polycentric metropolises (London, Milan, Stockholm).

### 2.2 Sample Point Generation

For each city, sample points are generated along the **pedestrian road network** using OSMnx/PyRosm. Points are placed at regular spacing on walkable edges, then **snapped to the nearest OSM node** via the ISO4App API, which also computes the 15-minute isochrone. This ensures every sample point has a valid walkable catchment area.

```
generate_points_parallel.py        # grid of candidate points per city
enrich_with_walking_isochrones.py  # ISO4App: snap + isochrone per point
```

### 2.3 Satellite Imagery — Sentinel-2

10 m/px Sentinel-2 Level-2A imagery (4 seasonal composites, cloud-masked) is downloaded via Google Earth Engine for each city bounding box.

- Resolution: **10 m/px**
- Crop: **224 × 224 px** centred on each sample point (= 2.24 × 2.24 km)
- Stored with ±40 px jitter padding (304 × 304) for augmentation

```
get_imgs_sentinel_2.py
```

### 2.4 Elevation Data — DEM

SRTM Digital Elevation Model at 30 m, resampled to 10 m to align with Sentinel-2. Normalised globally to [0, 1] using the fixed range [−50 m, 3000 m].

```
get_city_dem.py
```

### 2.5 Land Use Segmentation — ESA WorldCover

ESA WorldCover 2021 at 10 m resolution provides an 8-class land-use map per pixel (trees, shrubland, grassland, cropland, built-up, bare, water, wetland). One-hot encoded into 8 binary channels.

```
get_segmentation_esa.py
```

### 2.6 OSM Tabular Features

15 numerical features are extracted per sample point from OpenStreetMap within a **1120 m radius buffer** (= 224 px × 10 m/px ÷ 2), matching the spatial extent of the image crop:

| #    | Feature               | Description                                       |
| ---- | --------------------- | ------------------------------------------------- |
| 0    | `road_length_total`   | Total road length (m)                             |
| 1    | `road_count`          | Number of road segments                           |
| 2    | `road_density`        | Road length / buffer area (m/m²)                  |
| 3    | `node_count`          | Total road nodes                                  |
| 4    | `intersection_count`  | Nodes with ≥3 connections                         |
| 5    | `building_count`      | Number of building footprints                     |
| 6    | `building_area_total` | Total footprint area (m²)                         |
| 7    | `building_coverage`   | Building area / buffer area                       |
| 8–13 | `poi_*_count`         | POIs by tag (amenity/shop/leisure/tourism/office) |
| 14   | `transport_count`     | Transport stops/stations                          |

Features are log1p-transformed then z-scored using training-split statistics.

```
download_osm_city_data.py
extract_osm_tabular_features.py
```

### 2.7 Walkability Target

The target is the **residential population reachable within a 15-minute walk** from each sample point, extracted from the ISO4App isochrone intersected with WorldPop population grids. Values are clamped to [0, 80 000], then **per-city z-scored in log1p space**:

```
target = (log1p(population) − city_mean) / city_std
```

This per-city normalisation removes the large between-city scale differences (e.g. Milan vs. Innsbruck) and forces the model to learn within-city relative patterns.

### 2.8 City-Level Train / Val / Test Split

Cities are assigned entirely to one partition (no city appears in two splits), creating a strict domain-shift evaluation:

| Split | Cities | Approx. samples |
| ----- | ------ | --------------- |
| Train | 56     | ~31 000         |
| Val   | 12     | ~6 500          |
| Test  | 12     | ~6 500          |

**Test cities:** Birmingham, Espoo, Helsinki, Innsbruck, Lisbon, Lyon, Marseille, Milan, Stavanger, Stockholm, Utrecht, Zurich.

The split is deterministic and stored in `ablation/data/city_split.json`.

---

## 3. Data Preprocessing & Cache

Raw GeoTIFF rasters are slow to read per-sample during training. A one-time preprocessing step extracts and stores all crops as memory-mapped NumPy arrays:

```
ablation/data/preprocess.py         # builds the full cache
ablation/data/compute_rgb_stats.py  # per-city RGB mean/std (optional)
```

**Cache layout** (`preprocessed_cache/`):

```
rgb.dat           (N, 3, 304, 304)  uint8    raw RGB
dem.dat           (N, 1, 304, 304)  float32  DEM normalised [0,1]
landuse.dat       (N, 1, 304, 304)  uint8    class 0-7
targets.npy       (N,)              float32  per-city z-score
tabular.npy       (N, 15)          float32  log1p z-scored OSM
city_log_mean.npy (N,)              per-sample city log-mean
city_log_std.npy  (N,)              per-sample city log-std
index.json        [{city, cluster, split, valid}, ...]
stats.json        normalisation constants
```

During training, `CachedDataset` reads only the requested array slices — no GeoTIFF I/O. A ±40 px random jitter crop (224 × 224 from the stored 304 × 304) provides spatial augmentation.

---

## 4. Model Architecture

### 4.1 Input Representation

Each sample is a **12-channel tensor** (C × 224 × 224):

```
Channels 0-2   : RGB (Sentinel-2, normalised per-city or ImageNet stats)
Channel  3     : DEM elevation [0, 1]
Channels 4-11  : Land-use one-hot (8 classes)
```

### 4.2 Backbones

Four image backbones are evaluated, all pretrained on ImageNet (except DINOv2):

| Backbone        | Params | Feature dim | Notes                          |
| --------------- | ------ | ----------- | ------------------------------ |
| ResNet-50       | 25M    | 2048        | Standard baseline              |
| EfficientNet-B3 | 12M    | 1536        | Efficient compound scaling     |
| ConvNeXt-Tiny   | 28M    | 768         | Modern pure-CNN                |
| DINOv2 ViT-B/14 | 86M    | 768         | Self-supervised, 14×14 patches |

For 12-channel input, the first convolutional layer is expanded by averaging pretrained RGB weights across the extra 9 channels. DINOv2 uses `img_size=224` (positional embeddings interpolated from native 518 px).

An `*_rgb` variant of each backbone uses only the 3 RGB channels with the original pretrained weights (no channel expansion) — used for the RGB-only ablation.

### 4.3 Single-Branch Baseline

The backbone produces a global feature vector which is fed through a regression MLP:

```
x_img (12ch) → Backbone → feat (D,) → MLP → scalar prediction
```

Trained without any tabular features. Serves as the image-only ceiling.

### 4.4 Tabular-Only Baseline

No image backbone. A standalone MLP processes the 15 OSM features:

```
x_tab (15,) → MLP(128→64→32→1) → scalar prediction
```

Quantifies how much the pedestrian topology alone can predict walkability — the lower bound for multimodal models to beat.

### 4.5 RGB-Only Ablation

Identical to the single-branch model but uses only the 3 RGB channels with an `*_rgb` backbone. Removes DEM and land-use channels to isolate the visual-only signal.

### 4.6 Dual-Branch Fusion

Two parallel branches — one for image, one for tabular — whose features are concatenated before the prediction head:

```
x_img → Backbone → feat_img (D,)  ─┐
x_tab → TabMLP   → feat_tab (64,)  ─┴→ concat → MLP → prediction
```

Simple but effective: the model can freely route signal from either modality.

### 4.7 Cross-Attention Fusion

The tabular embedding acts as a _query_ attending over the spatial image feature map (H × W patches from the backbone). This allows the network to focus on image regions that are contextually relevant given the local OSM profile:

```
x_img → Backbone (global_pool="") → feat_map (D × H × W)
x_tab → TabMLP → query (D,)
query × feat_map → cross-attention → context → MLP → prediction
```

Inspired by the dual-branch interpretable architecture in Doda et al. [1].

### 4.8 DANN — Domain-Adversarial Neural Network

Cross-city generalisation is the fundamental challenge. DANN [3] addresses it by adding a **gradient-reversal domain classifier** that forces the backbone to learn city-invariant features:

```
x_img → Backbone → feat
              ├──→ Regression head → prediction
              └──→ GradReverse → Domain classifier → city label
```

The domain classifier is trained to predict the city of origin; the gradient reversal makes the backbone _unable_ to encode city-specific information. Paired with DINOv2 — which already produces domain-robust features due to self-supervised pretraining — this gives the best overall results.

A warmup phase (first N epochs) keeps the backbone frozen while the regression head stabilises, before adversarial training begins.

### 4.9 Other Tries: FiLM

FiLM (_Feature-wise Linear Modulation_) [2] modulates the intermediate activations of the image backbone conditioned on the tabular vector:

```
x_tab → γ, β generators
x_img → Backbone(layer k) → FiLM(γ, β) → Backbone(layer k+1) → ... → prediction
```

This allows the tabular features to dynamically rescale and shift each feature map channel. Evaluated with EfficientNet-B3 but underperformed the simpler dual-branch model, likely because modulation at intermediate layers introduces training instability at this scale.

---

## 5. Training Setup

| Hyperparameter          | Value                                        |
| ----------------------- | -------------------------------------------- |
| Loss                    | Huber (δ = 1.0)                              |
| Optimiser               | AdamW                                        |
| LR                      | 1 × 10⁻⁴                                     |
| LR scheduler            | ReduceLROnPlateau (patience 5)               |
| Batch size              | 64                                           |
| Early stopping patience | 5 (single/RGB-only) · 7 (multimodal)         |
| Warmup epochs           | 3 (DINOv2 models)                            |
| Augmentation            | ±40 px random crop, horizontal/vertical flip |
| Target normalisation    | Per-city z-score in log1p space              |
| Hardware                | NVIDIA A100 (Vast.ai)                        |

Training is managed by `ablation/training/trainer.py`. All runs are logged to TensorBoard and optionally W&B.

---

## 6. Results

### 6.1 Quantitative Comparison

All metrics computed on the held-out test cities (12 cities, never seen during training or validation). Population values are denormalised back to absolute counts before metric computation.

| Model                         |     MAE ↓ |    RMSE ↓ |      R² ↑ | MAPE ↓ |
| ----------------------------- | --------: | --------: | --------: | -----: |
| **DANN + DINOv2 ViT-B/14**    | **4 610** | **8 476** | **0.687** |  1 449 |
| Dual-Branch ConvNeXt-Tiny     |     5 075 |     9 090 |     0.640 |  1 118 |
| CrossAttn EfficientNet-B3     |     5 208 |     9 926 |     0.571 |  1 211 |
| Tabular-Only MLP _(baseline)_ |     5 897 |    10 851 |     0.488 |  1 143 |
| FiLM EfficientNet-B3 _(†)_    |     6 229 |    12 238 |     0.348 |    799 |
| RGB-Only EfficientNet-B3      |     8 053 |    13 447 |     0.213 |  7 316 |
| Single EfficientNet-B3        |     8 285 |    13 965 |     0.151 |  9 233 |
| RGB-Only DINOv2 ViT-B/14      |     8 360 |    13 992 |     0.148 |  8 510 |
| Single DINOv2 ViT-B/14        |     8 384 |    14 228 |     0.119 |  6 934 |
| Single ConvNeXt-Tiny          |     8 222 |    14 053 |     0.141 |  4 276 |

_(†) FiLM results from an earlier training run; not fully tuned._

**Key observations:**

- All image+tabular models beat the tabular-only baseline, confirming that satellite imagery adds complementary signal (+15% to +41% R²).
- Image-only models plateau at R² ≈ 0.1–0.2 regardless of backbone quality, confirming that walkability is fundamentally a network-topology task, not a visual one.
- DANN + DINOv2 is the clear winner (+22% MAE reduction vs. tabular-only), showing that domain-shift adaptation and strong visual pretraining are mutually beneficial.
- Counterintuitively, RGB-only models marginally outperform 12-channel single-branch models, suggesting the extra DEM/land-use channels add noise without the tabular branch to contextualise them.

### 6.2 Per-Cluster MAE

Samples are pre-clustered into 5 urban-morphology groups. Cluster 2 (dense/atypical urban forms) is the hardest for all models; DANN notably reduces its error.

| Model                     |    C0 |    C1 |        C2 |    C3 |    C4 |
| ------------------------- | ----: | ----: | --------: | ----: | ----: |
| DANN + DINOv2             | 4 545 | 4 915 | **5 527** | 4 691 | 3 341 |
| Dual ConvNeXt-Tiny        | 4 529 | 5 369 |     7 149 | 4 968 | 3 310 |
| CrossAttn EfficientNet-B3 | 3 916 | 5 562 |     7 414 | 5 297 | 3 790 |
| Tabular-Only              | 4 561 | 6 544 |     6 518 | 6 757 | 5 051 |
| FiLM EfficientNet-B3      | 4 639 | 6 977 |     7 530 | 7 011 | 4 915 |
| RGB-Only EfficientNet-B3  | 6 903 | 9 230 |    10 243 | 8 098 | 5 690 |
| Single EfficientNet-B3    | 7 105 | 9 216 |    10 501 | 8 497 | 6 015 |
| RGB-Only DINOv2           | 6 801 | 9 410 |    10 622 | 8 888 | 5 976 |
| Single DINOv2             | 6 818 | 9 154 |    10 900 | 8 999 | 5 962 |
| Single ConvNeXt-Tiny      | 6 792 | 9 249 |    10 694 | 8 460 | 5 811 |

DANN's advantage is particularly large on Cluster 2 (−23% vs. Dual ConvNeXt), the most domain-shifted urban type.

---

## 7. Visualizations

### 7.1 City Prediction Heatmaps

Sliding-window inference over the full city raster. Each grid point uses **real per-point OSM features** (extracted on-the-fly from the city's `.gpkg` files) and the city-specific z-score denormalisation. Bologna (training city) and Milan (test city) are shown side by side.

<table>
<tr>
<th>Model</th><th>Bologna (train) — Predicted</th><th>Milan (test) — Predicted</th>
</tr>
<tr>
<td>DANN DINOv2</td>
<td><img src="img/heatmap_dann_bologna.png" width="340"/></td>
<td><img src="img/heatmap_dann_milan.png" width="340"/></td>
</tr>
<tr>
<td>Dual ConvNeXt</td>
<td><img src="img/heatmap_dual_convnext_bologna.png" width="340"/></td>
<td><img src="img/heatmap_dual_convnext_milan.png" width="340"/></td>
</tr>
<tr>
<td>CrossAttn EfficientNet</td>
<td><img src="img/heatmap_crossattn_bologna.png" width="340"/></td>
<td><img src="img/heatmap_crossattn_milan.png" width="340"/></td>
</tr>
<tr>
<td>RGB-Only EfficientNet</td>
<td><img src="img/heatmap_rgb_efficientnet_bologna.png" width="340"/></td>
<td><img src="img/heatmap_rgb_efficientnet_milan.png" width="340"/></td>
</tr>
</table>

### 7.2 GradCAM / Gradient Saliency

For CNN-based models (EfficientNet-B3, ConvNeXt-Tiny), standard GradCAM highlights which image regions drive each prediction. For DINOv2, GradCAM produces uniform maps due to attention averaging in transformers; **gradient saliency** (∂prediction/∂pixels) is used instead.

Note: in multimodal models, GradCAM visualises only the image branch's contribution. Since OSM tabular features dominate (~70% of signal), maps are best interpreted as "what visual context the model uses as a secondary signal".

<table>
<tr>
<th>DANN DINOv2 — best predictions</th><th>Dual ConvNeXt — best predictions</th><th>CrossAttn EfficientNet — best predictions</th>
</tr>
<tr>
<td><img src="img/gradcam_dann.png" width="280"/></td>
<td><img src="img/gradcam_dual_convnext.png" width="280"/></td>
<td><img src="img/gradcam_crossattn.png" width="280"/></td>
</tr>
</table>

<table>
<tr>
<th>RGB-Only EfficientNet-B3 — best predictions (image-only model: maps are fully informative)</th>
</tr>
<tr>
<td><img src="img/gradcam_rgb_efficientnet.png" width="600"/></td>
</tr>
</table>

### 7.3 Cross-Attention Maps

For the CrossAttn model, the tabular query attends over the 7×7 spatial feature grid. Five sample points spread across the population distribution (10th → 90th percentile) are shown for both cities.

<table>
<tr>
<th>Bologna — cross-attention (5 patches)</th><th>Milan — cross-attention (5 patches)</th>
</tr>
<tr>
<td><img src="img/attention_bologna.png" width="400"/></td>
<td><img src="img/attention_milan.png" width="400"/></td>
</tr>
</table>

### 7.4 SHAP Feature Importance

SHAP values computed on the tabular branch of the best model (DANN + DINOv2) reveal which OSM features drive predictions most.

<table>
<tr>
<td><img src="img/shap_feature_importance.png" width="420"/></td>
<td><img src="img/shap_beeswarm.png" width="420"/></td>
</tr>
</table>

### 7.5 UMAP Embeddings

UMAP projections of the backbone feature space, coloured by population density. Well-separated clusters by population indicate the backbone has encoded walkability-relevant features.

<table>
<tr>
<th>DANN DINOv2</th><th>Dual ConvNeXt</th><th>RGB-Only EfficientNet</th>
</tr>
<tr>
<td><img src="img/umap_dann.png" width="270"/></td>
<td><img src="img/umap_dual_convnext.png" width="270"/></td>
<td><img src="img/umap_rgb_efficientnet.png" width="270"/></td>
</tr>
</table>

---

## 8. Reproduction

### 8.1 Environment Setup

```bash
git clone <repo>
cd OptimalAccessPointPrediction/ablation
pip install -r requirements.txt
# For MPS (Apple Silicon): standard pip install torch torchvision
```

### 8.2 Dataset Pipeline

Run each step in order (city TIF files must be downloaded first):

```bash
# 1. Generate sample points on the pedestrian network
python generate_points_parallel.py

# 2. Enrich with ISO4App 15-min walk isochrones + snap to OSM nodes
python enrich_with_walking_isochrones.py

# 3. Download Sentinel-2 imagery, DEM, ESA land-use
python get_imgs_sentinel_2.py
python get_city_dem.py
python get_segmentation_esa.py

# 4. Extract OSM tabular features (15 features, 1120m buffer)
python extract_osm_tabular_features.py

# 5. Preprocess into training cache (one-time, ~20 min on SSD)
python ablation/data/preprocess.py \
    --json data/final_clustered_samples.json \
    --base imgs/ \
    --out  preprocessed_cache/
```

### 8.3 Training

```bash
cd ablation/

# Train a single model
python scripts/train_dann_dinov2.py

# Train all models sequentially
python run_all.py \
    --cache /path/to/preprocessed_cache \
    --json  /path/to/final_clustered_samples.json \
    --base  /path/to/dataset
```

### 8.4 Evaluation & Visualisation

```bash
# City heatmaps + attention maps for all models (Bologna + Milan)
python run_all_visualize.py

# Attention maps only (fast — skips sliding-window heatmap)
python run_all_visualize.py --attention-only --model crossattn_efficientnet_b3

# GradCAM / saliency
python visualize_gradcam.py \
    --cache ../preprocessed_cache \
    --run dann_dinov2_vitb14 --model dann --backbone dinov2_vitb14 \
    --checkpoint ../outputs3/dann_dinov2_vitb14/best_model.pth

# SHAP feature importance
python visualize_shap.py --run dann_dinov2_vitb14 --model dann --backbone dinov2_vitb14

# UMAP embeddings
python visualize_umap.py --run dann_dinov2_vitb14 --model dann --backbone dinov2_vitb14
```

---

## 9. References

[1] S. Doda, M. Kahl, K. Ouan, I. Obadic, Y. Wang, H. Taubenböck, X.X. Zhu, _"Interpretable deep learning for consistent large-scale urban population estimation using Earth observation data"_, Int. Journal of Applied Earth Observation and Geoinformation, 128 (2024) 103731. https://doi.org/10.1016/j.jag.2024.103731

[2] E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville, _"FiLM: Visual Reasoning with a General Conditioning Layer"_, AAAI 2018. https://arxiv.org/abs/1709.07871

[3] Y. Ganin, E. Ustun, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, V. Lempitsky, _"Domain-Adversarial Training of Neural Networks"_, JMLR 17(1) (2016). https://arxiv.org/abs/1505.07818

[4] M. Oquab et al., _"DINOv2: Learning Robust Visual Features without Supervision"_, TMLR 2024. https://arxiv.org/abs/2304.07193

[5] M. Tan, Q.V. Le, _"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"_, ICML 2019. https://arxiv.org/abs/1905.11946

[6] Z. Liu et al., _"A ConvNet for the 2020s"_, CVPR 2022. https://arxiv.org/abs/2201.03545
