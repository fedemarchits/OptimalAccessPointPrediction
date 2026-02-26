# OptimalAccessPointPrediction

> **Predicting residential population within a 15-minute walk from any GPS point in a European city — using satellite imagery, elevation, land use, and OpenStreetMap features.**

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline](#pipeline)
3. [Dataset](#dataset)
   - [Cities](#cities)
   - [Input Features](#input-features)
   - [Target Labels](#target-labels)
4. [Models](#models)
   - [Architectures](#architectures)
   - [Results](#results)
   - [Prediction Heatmaps](#prediction-heatmaps)
5. [Repository Structure](#repository-structure)
6. [Quick Start](#quick-start)
7. [References](#references)

---

## Overview

Given any GPS coordinate in a European city, can we estimate how many people live within a 15-minute walk — using only publicly available data?

This project builds an end-to-end pipeline to answer that question:

- **Data collection**: Sentinel-2 satellite imagery, Copernicus DEM elevation, OpenStreetMap land use and neighbourhood statistics, and population isochrones from the iso4app API — across **90 European cities**.
- **Dataset**: 38,134 labelled sample points, each paired with a 224×224 px multi-channel crop and 15 OSM tabular features.
- **Ablation study**: Three multi-modal deep learning architectures are compared, all built on EfficientNet-B3, differing only in how they fuse image and tabular information.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                       │
│                                                          │
│  Sentinel-2 RGB  ──┐                                     │
│  Copernicus DEM  ──┼──▶  City-level GeoTIFF stack        │
│  OSM Land Use    ──┘                                     │
│                                                          │
│  OSM Road/Building/POI density ──▶ K-Means (5 clusters)  │
│  iso4app API ──▶ 15-min walk isochrone + population      │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                       DATASET                            │
│                                                          │
│  38,134 samples across 80 cities                         │
│  Image  : 224×224 px crop (12 channels)                  │
│  Tabular: 15 OSM neighbourhood features                  │
│  Target : population within 15-min walk (regression)     │
│  Split  : 70/15/15 city-level train/val/test (seed=42)   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    ABLATION STUDY                         │
│                                                          │
│  Backbone: EfficientNet-B3 (pretrained, 12-ch input)     │
│                                                          │
│  ① Dual-Branch     — concat fusion                       │
│  ② Cross-Attention — tabular queries spatial map         │
│  ③ FiLM            — channel-wise scale + shift          │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

→ Full dataset documentation: [`dataset_creation/README.md`](dataset_creation/README.md)

### Cities

**90 major European cities** across 12 countries, selected for their shared urban structure and availability of free geospatial and census data.

| 🇮🇹 Italy | 🇫🇷 France | 🇬🇧 UK | 🇧🇪 Belgium | 🇳🇱 Netherlands | 🇸🇪 Sweden |
|:-------:|:-------:|:----:|:-------:|:-----------:|:------:|
| 9 cities | 10 cities | 9 cities | 7 cities | 7 cities | 7 cities |

| 🇨🇭 Switzerland | 🇦🇹 Austria | 🇫🇮 Finland | 🇵🇹 Portugal | 🇳🇴 Norway | 🇩🇰 Denmark + 🇬🇷 Greece |
|:-----------:|:-------:|:-------:|:-------:|:-------:|:-------------------:|
| 7 cities | 6 cities | 6 cities | 6 cities | 5 cities | 3 cities |

### Input Features

Each sample consists of two modalities:

**Image — 12-channel crop (224×224 px at 10m resolution)**

| Channels | Source | Description |
|:--------|:-------|:-----------|
| 1–3 (RGB) | Sentinel-2 L2A | True colour, cloud-free summer median composite, census-year aligned |
| 4 (DEM) | Copernicus GLO-30 | Absolute elevation in metres, globally normalised to [0, 1] |
| 5–12 (Land Use) | OpenStreetMap | One-hot encoded: Residential, Commercial, Industrial, Retail, Public, Parks, Natural, Background |

**Tabular — 15 OSM features**

Road density, building density, POI count, and other neighbourhood statistics computed within a 250 m buffer around each sample point. Applied log1p + z-score normalisation using training-split statistics.

**Satellite imagery — Bologna and Nantes**

<table>
  <tr>
    <td align="center">
      <img src="img/Bologna_Italy_10m.png" width="380px"/>
      <br/><sub><b>Sentinel-2 capture — Bologna, Italy</b></sub>
    </td>
    <td align="center">
      <img src="img/Nantes_France_10m.png" width="380px"/>
      <br/><sub><b>Sentinel-2 capture — Nantes, France</b></sub>
    </td>
  </tr>
</table>

### Target Labels

Points are generated via K-Means clustering (5 clusters per city based on road/building/POI density), ensuring the dataset covers the full spectrum from dense historic centres to sparse suburbs. The target for each point is the **total resident population reachable within a 15-minute walk**, queried from the [iso4app](https://www.iso4app.net/) API using pedestrian isochrones. Targets are clamped to [0, 80,000] and log1p-normalised during training.

**Cluster maps — Bologna and Nantes**

<table>
  <tr>
    <td align="center">
      <img src="img/Bologna_Italy_clusters.png" width="380px"/>
      <br/><sub><b>5-cluster stratified sampling — Bologna, Italy</b></sub>
    </td>
    <td align="center">
      <img src="img/Nantes_France_clusters.png" width="380px"/>
      <br/><sub><b>5-cluster stratified sampling — Nantes, France</b></sub>
    </td>
  </tr>
</table>

---

## Models

→ Full ablation documentation and run guide: [`ablation/README.md`](ablation/README.md)

### Architectures

All three models share the same **EfficientNet-B3** backbone (pretrained on ImageNet, adapted for 12-channel input). They differ only in how image and tabular branches are fused.

---

**① Dual-Branch — Concat Fusion**

The simplest multi-modal baseline. Image features (global average pool) and tabular features (small MLP) are concatenated before a shared regression head.

```
image   → EfficientNet-B3 → GAP → (B, 1536) ──┐
                                                ├─▶ concat(1600) → Head → scalar
tabular → MLP(15→64)            → (B,   64) ──┘
```

---

**② Cross-Attention Fusion**

The tabular embedding acts as a **query** that attends over the 7×7 spatial feature map of EfficientNet, letting OSM neighbourhood context dynamically focus the model on the most informative image regions.

```
image   → EfficientNet-B3 (no pool) → (B, 49, 1536)   ← keys / values
tabular → MLP(15→64) → proj(256)    → (B,  1,  256)   ← query

Cross-Attention → (B, 256)
concat( GAP(1536) | attended(256) | tab(64) ) → Head → scalar
```

---

**③ FiLM Conditioning**

Tabular features predict per-channel **scale (γ) and shift (β)** applied directly to the EfficientNet feature map before pooling — controlling *which channels matter* rather than *where to look*.

```
image   → EfficientNet-B3 (no pool) → (B, 1536, 7, 7)
tabular → MLP(15→64) → FiLMGen → γ (B, 1536),  β (B, 1536)

modulated = γ ⊙ spatial + β  →  GAP  →  (B, 1536)
concat(1600) → Head → scalar
```

---

### Results

Training: AdamW (lr=1e-4), Huber loss (δ=1), ReduceLROnPlateau, early stopping (patience=10).
38,134 samples across 80 European cities.

#### Test Set

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Epochs trained |
|:------|------:|-------:|-----:|---------------:|
| **Dual-Branch (concat)** | **6,077** | **12,915** | **0.379** | 43 |
| Cross-Attention | 6,287 | 13,020 | 0.369 | 17 |
| FiLM | 7,207 | 14,710 | 0.195 | 21 |

#### Validation (best epoch)

| Model | Best Val MAE | Best Val R² |
|:------|------------:|------------:|
| Dual-Branch | 3,944 | 0.672 |
| Cross-Attention | 4,059 | **0.691** |
| FiLM | 4,258 | 0.608 |

**Key takeaways:**
- **Dual-Branch** achieves the best test MAE and R², showing that simple concat fusion generalises well at this dataset scale.
- **Cross-Attention** reaches the highest validation R² (0.691) and converges in just 17 epochs — roughly 2.5× faster than Dual-Branch. The gap on the test set may reflect checkpoint selection on val loss rather than R².
- **FiLM** shows the weakest test generalisation. Generating 3,072 FiLM parameters from a 64-dim embedding is a high-dimensional mapping that may be too unconstrained for this dataset size.

### Prediction Heatmaps

Dense sliding-window inference (stride=16 px) across the full city satellite image. Colour intensity = predicted population within 15-minute walk.

**Cross-Attention model — Bologna, Italy**
![Cross-Attention heatmap](ablation/assets/heatmap_crossattn.png)

**FiLM model — Bologna, Italy**
![FiLM heatmap](ablation/assets/heatmap_film.png)

---

## Repository Structure

```
OptimalAccessPointPrediction/
│
├── dataset_creation/               ← Data collection pipeline
│   ├── city_bounding_box_generator.py
│   ├── point_clusters_generator.py
│   ├── population_isochrones_generator.py
│   ├── sentinel2_images_generator.py
│   └── README.md                   ← Full dataset documentation
│
├── ablation/                       ← Model training and evaluation
│   ├── configs/                    ← Experiment configs
│   ├── data/                       ← Dataset, preprocessing, dataloaders
│   ├── models/                     ← Backbones, fusion architectures
│   ├── training/                   ← Trainer, metrics, early stopping
│   ├── scripts/                    ← One script per experiment
│   ├── visualize.py                ← Heatmap and attention map generation
│   ├── assets/                     ← Result images
│   └── README.md                   ← Ablation results + full run guide
│
├── data/
│   └── cities_bboxes_major_europe.json
│
├── img/                            ← Sample satellite and cluster images
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/your-username/OptimalAccessPointPrediction.git
cd OptimalAccessPointPrediction

# Data pipeline dependencies
pip install -r requirements.txt

# Model training dependencies
pip install -r ablation/requirements.txt
```

For dataset generation, see [`dataset_creation/README.md`](dataset_creation/README.md).

For training (recommended on a GPU cloud instance), see [`ablation/README.md`](ablation/README.md).

---

## References

Inspired by:

> *Predicting human mobility flows in cities using deep learning on satellite imagery*
> Nature Communications, 2025.
> [https://www.nature.com/articles/s41467-025-65373-z](https://www.nature.com/articles/s41467-025-65373-z)

**Data sources:**
- [Sentinel-2 L2A](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) — ESA / Copernicus
- [Copernicus DEM GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) — via Microsoft Planetary Computer
- [OpenStreetMap](https://www.openstreetmap.org/) — via `osmnx`
- [iso4app](https://www.iso4app.net/) — walking isochrones and population data
