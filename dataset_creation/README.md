# 🏗️ Dataset Creation Pipeline

This project builds a comprehensive dataset to train an AI model capable of predicting **Optimal Access Points**—locations that maximize population reachability within 15 minutes. The dataset creation is divided into three distinct phases: **Spatial Definition**, **Feature Extraction**, and **Target Estimation**.

---

## 📍 Phase 1: Spatial Definition (City Bounding Boxes)

**Goal:** Define consistent geographic boundaries for 80 major European cities to standardize the study areas.

Using the `osmnx` library, we query the OpenStreetMap Nominatim API to fetch the official administrative boundaries for a list of diverse cities (e.g., Rome, Paris, Stockholm).

- **Input:** List of "City, Country" strings.
- **Process:**
  1.  **Geocoding:** `ox.geocode_to_gdf(city_name)` retrieves the polygon of the city.
  2.  **Bounding Box Extraction:** We extract the `total_bounds` (North, South, East, West) to create a standardized rectangular analysis window.
- **Output:** A JSON file `cities_bboxes_major_europe.json` containing the precise bounding coordinates for 80 cities.

---

## 🔍 Phase 2: Stratified Sampling & Feature Extraction

**Goal:** Generate a diverse, balanced set of ~40,000 candidate points that represent the full spectrum of urban environments, from dense city centers to sparse suburbs and empty areas.

We utilize a custom "Safe-Sampling" script to process each city without hitting API limits.

### 1. Adaptive Grid Generation

Instead of random sampling, we generate a uniform grid of candidate points (approx. 2000 per city).

- **Logic:** The script calculates an adaptive step size (in meters) based on the city's bounding box area to ensuring consistent sampling density regardless of city size.

### 2. "Safe" Chunked Downloading

To handle massive data volumes without crashing, the script divides the city into **5km spatial chunks**. It downloads OpenStreetMap (OSM) data piecemeal, sleeping between requests to respect API rate limits.

### 3. Metric Calculation (The Features $X$)

For every candidate point in the grid, we calculate three core density metrics within a **250m radius**:

- **Road Density:** Connectivity level ($m/m^2$).
- **Building Density:** Urbanization level ($m^2/m^2$).
- **POI Density:** Activity level (count of amenities, shops, offices).

### 4. Clustering & Stratified Sampling

To avoid bias (e.g., only selecting city centers), we perform **K-Means Clustering ($k=5$)** on the calculated metrics.

- **Cluster 0:** Empty/Sea/Wilderness (Zero density).
- **Clusters 1-2:** Low-density suburbs/residential.
- **Clusters 3-4:** High-density urban cores.

We then randomly sample **100 points per cluster**, ensuring our training data includes "bad" locations (sea/empty fields) as well as "good" ones. This results in **500 representative points per city**.

---

## 🎯 Phase 3: Target Variable Estimation (Ground Truth)

**Goal:** Determine the "Optimality" of each selected point by calculating the actual reachable population.

For each of the ~40,000 selected points, we perform a query to the **iso4app API**:

1.  **Isochrone Generation:** We compute the specific geographic polygon reachable within a **15-minute drive/ride** from the point.
2.  **Population Intersection:** We intersect this isochrone polygon with high-resolution population density rasters.
3.  **Target Label ($Y$):** The final value associated with each point is the **Total Reachable Population**.

**Final Dataset Structure:**
| Latitude | Longitude | Road Density ($X_1$) | Building Density ($X_2$) | POI Density ($X_3$) | **Reachable Pop ($Y$)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 41.902 | 12.496 | 0.45 | 0.82 | 120 | **45,300** |
| ... | ... | ... | ... | ... | ... |
