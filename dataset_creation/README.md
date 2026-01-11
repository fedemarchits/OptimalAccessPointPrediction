# Dataset

## Table of Contents

- [Introduction](#introduction)
- [Dataset Input Creation](#dataset-input-creation)

## Introduction

### 🏙️ Cities

We chose to focus exclusively on Europe for our case study for the following reasons:

- European cities share a **distinctive urban structure**. We limited data collection to cities with this characteristic, excluding those with fundamentally different layouts (such as American and Chinese cities);

- **Abundant free geospatial data** is available for Europe.

Moreover, we selected countries for which population data is available via [**iso4app**](https://www.iso4app.net/).

These dataset considers **90 major cities** in **12 different European Countries**. Below the list of the cities considered.

| 🇮🇹 **Italy** (9) | 🇫🇷 **France** (10) | 🇦🇹 **Austria** (6) | 🇬🇧 **United Kingdom** (9) |
| ---------------- | ------------------ | ------------------ | ------------------------- |
| Rome             | Paris              | Vienna             | London                    |
| Milan            | Marseille          | Graz               | Birmingham                |
| Naples           | Lyon               | Linz               | Glasgow                   |
| Turin            | Toulouse           | Salzburg           | Leeds                     |
| Palermo          | Nice               | Innsbruck          | Liverpool                 |
| Bologna          | Nantes             | Klagenfurt         | Manchester                |
| Florence         | Montpellier        |                    | Bristol                   |
| Bari             | Strasbourg         |                    | Sheffield                 |
| Catania          | Bordeaux           |                    | Edinburgh                 |
|                  | Lille              |                    |

| 🇧🇪 **Belgium** (7) | 🇩🇰 **Denmark** (5) | 🇫🇮 **Finland** (6) |
| ------------------ | ------------------ | ------------------ |
| Brussels           | Copenhagen         | Helsinki           |
| Antwerp            | Aarhus             | Espoo              |
| Ghent              | Odense             | Tampere            |
| Charleroi          | Aalborg            | Vantaa             |
| Liège              | Esbjerg            | Oulu               |
| Bruges             |                    | Turku              |
| Namur              |                    |                    |

| 🇬🇷 **Greece** (6) | 🇳🇱 **Netherlands** (7) | 🇳🇴 **Norway** (5) |
| ----------------- | ---------------------- | ----------------- |
| Athens            | Amsterdam              | Oslo              |
| Thessaloniki      | Rotterdam              | Bergen            |
| Patras            | The Hague              | Trondheim         |
| Heraklion         | Utrecht                | Stavanger         |
| Larissa           | Eindhoven              | Tromsø            |
| Volos             | Tilburg                |                   |
|                   | Groningen              |                   |

| 🇵🇹 **Portugal** (6) | 🇸🇪 **Sweden** (7) | 🇨🇭 **Switzerland** (7) |
| ------------------- | ----------------- | ---------------------- |
| Lisbon              | Stockholm         | Zurich                 |
| Porto               | Gothenburg        | Geneva                 |
| Vila Nova de Gaia   | Malmö             | Basel                  |
| Amadora             | Uppsala           | Lausanne               |
| Braga               | Västerås          | Bern                   |
| Coimbra             | Örebro            | Winterthur             |
|                     | Linköping         | Lucerne                |

## Dataset Input Creation

What we want to achieve for the input is a **3D tensor** with:

- **6 channels**;
- **H x W** pixels depending on the city;
- **10m spatial resolution**.

Each channel is explained below:

| Channel                      | Source                                        | Resolution | Classes/Range                                                                |
| ---------------------------- | --------------------------------------------- | ---------- | ---------------------------------------------------------------------------- |
| **RGB** _3 channels_         | Sentinel-2 (Copernicus)                       | 10m        | True color (B4,B3,B2)                                                        |
| **Height** _1 channel_       | TanDEM-X DEM                                  | 12m        | Elevation (m)                                                                |
| **Segmentation** _1 channel_ | WorldCover (ESA) or Dynamic LULC (Copernicus) | 10m        | 11 classes: forest, cropland, built-up, water, park, industrial equiv., etc. |
| **Vegetation** _1 channel_   | HR-VPP NDVI (Copernicus)                      | 10m        | NDVI (-1 to 1)                                                               |

### RGB Images (Sentinel-2 | Copernicus)

The RGB images we got are from **Sentinel-2** satellite, they are at a **10m resolution**. More specifically the script in charge of downloading images can be founder here [`get_RGB_sentinel.py`](get_RGB_sentinel.py).

**Key Features & Methodology:**

- **Census-Aligned Temporal Synchronization:**

  The script dynamically adjusts the search window for each city to match the specific year of its national census (e.g., _Summer 2018_ for Switzerland, _Summer 2023_ for Italy). This ensures the physical urban fabric observed in the images corresponds accurately to the demographic data used in the analysis. More specifically these are the population data available:

  | Country            | Population Source  | Census Year | Satellite Image Year\* |
  | :----------------- | :----------------- | :---------- | :--------------------- |
  | **Austria** 🇦🇹     | HDX                | 2020        | 2020                   |
  | **Belgium** 🇧🇪     | STATBEL            | 2020        | 2020                   |
  | **Denmark** 🇩🇰     | DST                | 2024        | 2023                   |
  | **Finland** 🇫🇮     | STATISTICS FINLAND | 2022        | 2022                   |
  | **France** 🇫🇷      | INSEE census       | 2020        | 2020                   |
  | **Greece** 🇬🇷      | GEODATA.GOV.GR     | 2020        | 2020                   |
  | **Italy** 🇮🇹       | ISTAT census       | 2024        | 2023                   |
  | **Netherlands** 🇳🇱 | CBS                | 2022        | 2022                   |
  | **Norway** 🇳🇴      | SSB                | 2024        | 2023                   |
  | **Portugal** 🇵🇹    | INE                | 2018        | 2018                   |
  | **Sweden** 🇸🇪      | SCB                | 2020        | 2020                   |
  | **Switzerland** 🇨🇭 | STATPOP            | 2018        | 2018                   |

  > \* _Note: For 2024 census releases (Italy, Denmark, Norway), satellite imagery from Summer 2023 is used as the closest complete summer season._

You can check directly on [**iso4app**](https://www.iso4app.net/) website for specifications about the census data we used.

- **Seasonal Consistency (European Summer):**

  - All images are restricted to the months of **June through September**.
  - This "Summer Window" is critical for Northern Europe (e.g., Norway, Finland) to ensure scenes are **snow-free**, have maximum solar elevation (minimizing long shadows in urban canyons), and show vegetation at peak greenness for consistent spectral analysis.

- **Cloud-Free Mosaicing:**

  - Queries the **Sentinel-2 L2A** collection via the STAC API, strictly filtering for scenes with **<95% cloud cover**, basically keeping only those images where a portion of the city is visible.
  - When multiple passes are available, it computes a **median pixel composite**. This technique effectively removes transient clouds, shadows, and artifacts that might appear in a single flyover, resulting in a clean, obstruction-free image.

- **Post-Processing:**
  - **Bands:** Extracts RGB bands (10m resolution).
  - **Normalization:** Clips and scales raw reflectance values to standard 8-bit RGB integers.
  - **Export:** Saves the final output as georeferenced TIFFs (`EPSG:3857`) ready for computer vision tasks.

### Height Data (TanDEM-X DEM)

### Segmentation Data (WorldCover (ESA))

### Vegetation (HR-VPP NDVI | Copernicus)

## Ground Truth Creation

# 🏙️ Urban Analysis & Isochrone Enrichment Pipeline

This pipeline automates the extraction of urban data across major European cities. It progresses from defining geographical boundaries to generating stratified sampling points based on urban density, and finally enriching those points with walking isochrones and population data.

## Pipeline Overview

The workflow consists of three sequential stages:

1.  **Scope Definition**: Extracts bounding boxes for target cities.
2.  **Sampling & Clustering**: Generates points, calculates urban density metrics (roads, buildings, POIs), clusters them to ensure diverse representation (e.g., city center vs. suburbs), and selects a stratified sample.
3.  **Enrichment**: Queries external APIs to calculate 15-minute walking areas and population counts for the selected points.

---

## 📂 Script Details

### 1. City Bounding Box Generator

**Script:** `city_bounding_box_generator.py`

This script initializes the project by defining the geographical scope. It iterates through a the cities listed earlier and uses OpenStreetMap (OSMnx) to retrieve their geographical boundaries.

- **Input**: A hardcoded list of city names (e.g., "Rome, Italy", "Paris, France").
- **Logic**:
  - Geocodes each city name to a GeoDataFrame.
  - Extracts the `total_bounds` (min/max lat/lon).
  - Formats coordinates into a polygon structure.
- **Output**: `cities_bboxes_major_europe.json` containing the bounding box coordinates for each successfully processed city.

### 2. Parallel Sampling & Clustering

**Script:** `point_clusters_generator.py`

This is the core processing engine. It moves from raw bounding boxes to a set of intelligent, representative points for analysis. It uses parallel processing to handle large datasets efficiently.

- **Input**: `cities_bboxes_major_europe.json`
- **Logic**:
  - **Adaptive Grid**: Generates a grid of candidate points within the city bounding box.
  - **Parallel Data Fetching**: Uses a `ThreadPoolExecutor` to download OSM features (Roads, Buildings, POIs) in 3km chunks to avoid API timeouts.
  - **Metric Calculation**: For every grid point, it calculates density within a 250m buffer:
    - _Road Density_: Total length of road network.
    - _Building Density_: Total footprint area of buildings.
    - _POI Density_: Count of amenities, shops, and offices.
  - **K-Means Clustering**: Normalizes these metrics and groups points into **5 clusters** (representing different urban fabrics, e.g., "Dense Historic Center", "Residential", "Sparse Industrial").
  - **Stratified Sampling**: Selects a fixed number of random points (default: 100) from each cluster to ensuring the final dataset represents the full diversity of the city.
- **Output**: `final_clustered_samples.json` (A JSON list of sampled points with their cluster IDs and density metrics).
- **Visualization**: Saves cluster maps to `./imgs/cluster_images/` for visualization and verification.

### 3. Isochrone Enrichment

**Script:** `population_isochrones_generator.py`

This script adds human-centric mobility data to the clustered points using the [**iso4app API**](https://www.iso4app.net/). It operates "in-place," meaning it updates the existing dataset.

- **Input**: `final_clustered_samples.json`, the list of points previously computed
- **Logic**:
  - **Isochrone Calculation**: Queries the API for a 15-minute walking polygon (pedestrian mobility) starting from the point's coordinates.
  - **Population Lookup**: Uses the generated polygon to query the API for the total population living within that specific walking area.
  - **Error Handling**: Includes retry logic and periodic saving (every 50 points) to prevent data loss.
- **Output**: Updates `final_clustered_samples.json` by appending:
  - `walking_isochrone`: The GeoJSON geometry of the walkable area.
  - `population_15min_walk`: The integer count of people inside that area.
  - `approximation_meters`: Distance from the query point to the nearest walkable street.
