import matplotlib
matplotlib.use('Agg') 

import json
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from math import radians, cos, sqrt
import warnings
import os
import time
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_FILE = "./data/cities_bboxes_major_europe.json"
OUTPUT_FILE = "./data/final_clustered_samples.json"
SAMPLES_PER_CLUSTER = 100   
N_CLUSTERS = 5
MIN_CANDIDATES = 2000
TARGET_STEP_METERS = 450
MAX_WORKERS = 4  # OPTIMIZED: Matches your 4-slot limit

# --- SETTINGS ---
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = "./cache"
ox.settings.user_agent = "CityAnalysis_ResearchProject/1.0"
ox.settings.timeout = 60
warnings.filterwarnings("ignore")

def parse_bbox_from_json(bbox_coords):
    lats = [p[0] for p in bbox_coords]
    lons = [p[1] for p in bbox_coords]
    return min(lons), min(lats), max(lons), max(lats)

def generate_adaptive_grid(minx, miny, maxx, maxy, min_candidates=2000, target_step_meters=450):
    lat_center = (miny + maxy) / 2
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 40075000 * cos(radians(lat_center)) / 360
    
    width_m = (maxx - minx) * meters_per_deg_lon
    height_m = (maxy - miny) * meters_per_deg_lat
    total_area_m2 = width_m * height_m
    
    estimated_points_at_target = total_area_m2 / (target_step_meters ** 2)
    
    if estimated_points_at_target < min_candidates:
        adaptive_step_m = sqrt(total_area_m2 / min_candidates)
    else:
        adaptive_step_m = target_step_meters

    step_x_deg = adaptive_step_m / meters_per_deg_lon
    step_y_deg = adaptive_step_m / meters_per_deg_lat
    
    x_coords = np.arange(minx, maxx, step_x_deg)
    y_coords = np.arange(miny, maxy, step_y_deg)
    points = [Point(x, y) for x in x_coords for y in y_coords]
    return gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# --- OPTIMIZED CHUNKING: 2900m STEP ---
def get_chunk_points(minx, miny, maxx, maxy, step_meters=2900):
    """
    Generates download centers with 2900m spacing.
    This creates slight overlap for 3000m (radius=1500) tiles.
    """
    lat_center = (miny + maxy) / 2
    
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * cos(radians(lat_center))
    
    step_x = step_meters / meters_per_deg_lon
    step_y = step_meters / meters_per_deg_lat
    
    download_points = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            download_points.append((y, x))
            y += step_y
        x += step_x
    return download_points

# --- HELPER WORKER FUNCTIONS ---

def fetch_road_chunk(coords):
    lat, lon = coords
    try:
        for attempt in range(3):
            try:
                # OPTIMIZED: dist=1500 (3km wide tile)
                G = ox.graph_from_point((lat, lon), dist=1500, network_type='drive', simplify=True)
                if len(G.nodes) > 0:
                    return G
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1)) 
                    continue
                raise e
            break
    except Exception:
        return None
    return None

def fetch_feature_chunk(args):
    coords, tags = args
    lat, lon = coords
    try:
        for attempt in range(3):
            try:
                # OPTIMIZED: dist=1500 (3km wide tile)
                gdf = ox.features_from_point((lat, lon), tags=tags, dist=1500)
                if not gdf.empty:
                    return gdf
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e
            break
    except Exception:
        return None
    return None

# --- PARALLEL DOWNLOADER FUNCTIONS ---

def download_roads_safe(download_points):
    print(f"      Downloading Roads ({len(download_points)} big chunks)...")
    all_graphs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_road_chunk, pt): pt for pt in download_points}
        
        for i, future in enumerate(as_completed(futures)):
            print(f"        [Roads] Chunk {i+1}/{len(download_points)}...", end='\r')
            result = future.result()
            if result is not None:
                all_graphs.append(result)
            if i % 10 == 0: time.sleep(0.1)

    print(f"\n      ✔ Got {len(all_graphs)} road chunks.")
    if not all_graphs: return gpd.GeoDataFrame()
    
    combined_G = nx.compose_all(all_graphs)
    combined_G = ox.convert.to_undirected(combined_G)
    _, edges_gdf = ox.graph_to_gdfs(combined_G)
    return edges_gdf[~edges_gdf.index.duplicated(keep='first')]

def download_features_safe(download_points, tags, label="Features"):
    print(f"      Downloading {label} ({len(download_points)} chunks)...")
    all_gdfs = []
    
    tasks = [(pt, tags) for pt in download_points]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_feature_chunk, t): t for t in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            print(f"        [{label}] Chunk {i+1}/{len(download_points)}...", end='\r')
            result = future.result()
            if result is not None:
                all_gdfs.append(result)
                
    print(f"\n      ✔ Got {len(all_gdfs)} {label} chunks.")
    if not all_gdfs: return gpd.GeoDataFrame()
    
    combined = pd.concat(all_gdfs)
    return combined[~combined.index.duplicated(keep='first')]

# --- METRIC CALCULATION ---

def calculate_full_metrics(gdf_grid, roads, buildings, pois):
    """Calculates Road, Building, and POI density"""
    if gdf_grid.empty: return gdf_grid
    
    utm_crs = gdf_grid.estimate_utm_crs()
    gdf_utm = gdf_grid.to_crs(utm_crs)
    gdf_buffers = gdf_utm.copy()
    
    gdf_buffers['geometry'] = gdf_buffers.geometry.buffer(250)
    buffer_area = np.pi * (250**2)

    # 1. Road Density
    if not roads.empty:
        roads_utm = roads.to_crs(utm_crs)
        joined_roads = gpd.sjoin(roads_utm, gdf_buffers, predicate='intersects', how='inner')
        if 'length' in roads_utm.columns:
            joined_roads['len'] = joined_roads['length']
        else:
            joined_roads['len'] = joined_roads.geometry.length
            
        road_sums = joined_roads.groupby('index_right')['len'].sum()
        gdf_grid['road_density'] = gdf_grid.index.map(road_sums).fillna(0) / buffer_area
    else:
        gdf_grid['road_density'] = 0

    # 2. Building Density
    if not buildings.empty:
        bld_utm = buildings.to_crs(utm_crs)
        bld_poly = bld_utm[bld_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        joined_bld = gpd.sjoin(bld_poly, gdf_buffers, predicate='intersects', how='inner')
        joined_bld['area'] = joined_bld.geometry.area
        bld_sums = joined_bld.groupby('index_right')['area'].sum()
        gdf_grid['building_density'] = gdf_grid.index.map(bld_sums).fillna(0) / buffer_area
    else:
        gdf_grid['building_density'] = 0

    # 3. POI Density (Count)
    if not pois.empty:
        pois_utm = pois.to_crs(utm_crs)
        joined_pois = gpd.sjoin(pois_utm, gdf_buffers, predicate='intersects', how='inner')
        poi_counts = joined_pois.groupby('index_right').size()
        gdf_grid['poi_density'] = gdf_grid.index.map(poi_counts).fillna(0)
    else:
        gdf_grid['poi_density'] = 0

    return gdf_grid

# --- FIXED FUNCTION: CORRECT CLUSTERING ---
def cluster_and_sample(gdf):
    if len(gdf) < N_CLUSTERS: return gdf
    
    # 1. Use ONLY density features (Removed coords/p.x/p.y)
    features_to_cluster = ['road_density', 'building_density', 'poi_density']
    
    valid_features = [f for f in features_to_cluster if f in gdf.columns and gdf[f].sum() > 0]
    
    if not valid_features:
        return gdf 
    
    scaler = MinMaxScaler()
    scaled_features = []
    
    # 2. Scale only the density data
    for feature in valid_features:
        f_data = gdf[feature].values.reshape(-1, 1)
        scaled_features.append(scaler.fit_transform(f_data))
    
    # 3. Stack features and Cluster
    final_matrix = np.hstack(scaled_features)
    gdf['cluster'] = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(final_matrix)
    
    # 4. Stratified Sampling
    samples = []
    for c in range(N_CLUSTERS):
        subset = gdf[gdf['cluster'] == c]
        if len(subset) >= SAMPLES_PER_CLUSTER:
            samples.append(subset.sample(n=SAMPLES_PER_CLUSTER, random_state=42))
        else:
            samples.append(subset)
    
    return pd.concat(samples)

def plot_city_clusters(gdf, city_name):
    safe_name = city_name.replace(", ", "_").replace(" ", "_")
    os.makedirs("./imgs/cluster_images", exist_ok=True)
    
    gdf_web = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    gdf_web.plot(column='cluster', ax=ax, cmap='tab10', markersize=30, 
                 legend=True, categorical=True, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
    except:
        pass
    
    ax.set_title(f"{city_name}\nClustered Sampling Points", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    plt.savefig(f"./imgs/cluster_images/{safe_name}_clusters.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    print("\n" + "="*70)
    print("FULL CITY SAMPLING GENERATOR (Parallel Mode 🚀)")
    print("="*70)
    
    os.makedirs("./cache", exist_ok=True)
    os.makedirs("./imgs/cluster_images", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_FILE}.")
        return

    cities = data.get("cities", [])
    all_results = []
    processed_cities = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"📂 Found existing data file: {OUTPUT_FILE}")
            existing_df = pd.read_json(OUTPUT_FILE)
            if not existing_df.empty and 'city' in existing_df.columns:
                all_results.append(existing_df)
                processed_cities = set(existing_df['city'].unique())
                print(f"✔ Resuming... Skipping {len(processed_cities)} cities already completed.")
                print(f"  (Skipped: {', '.join(list(processed_cities)[:3])}...)")
        except ValueError:
            print("⚠ Output file exists but appears corrupt or empty. Starting fresh.")
    
    cities_to_process = [c for c in cities if c['city'] not in processed_cities]
    print(f"🚀 {len(cities_to_process)} cities remaining to process.")
    
    pbar = tqdm(cities_to_process, unit="city")
    
    for entry in pbar:
        city_name = entry['city']
        pbar.set_description(f"Processing {city_name.split(',')[0]}")
        
        try:
            bbox_coords = entry['bbox']
            minx, miny, maxx, maxy = parse_bbox_from_json(bbox_coords)
            
            gdf_grid = generate_adaptive_grid(minx, miny, maxx, maxy, min_candidates=MIN_CANDIDATES)
            chunk_points = get_chunk_points(minx, miny, maxx, maxy)
            
            pbar.write(f"\n   Downloading data for {city_name}...")
            roads = download_roads_safe(chunk_points)
            buildings = download_features_safe(chunk_points, {'building': True}, "Buildings")
            pois = download_features_safe(chunk_points, {'amenity': True, 'shop': True, 'office': True, 'leisure': True}, "POIs")
            
            gdf_features = calculate_full_metrics(gdf_grid, roads, buildings, pois)
            gdf_sampled = cluster_and_sample(gdf_features)
            plot_city_clusters(gdf_sampled, city_name)
            
            gdf_sampled['city'] = city_name
            gdf_sampled['lat'] = gdf_sampled.geometry.y
            gdf_sampled['lon'] = gdf_sampled.geometry.x
            
            cols = ['city', 'cluster', 'lat', 'lon', 'road_density', 'building_density', 'poi_density']
            existing_cols = [c for c in cols if c in gdf_sampled.columns]
            
            new_df = gdf_sampled[existing_cols]
            all_results.append(new_df)

            pd.concat(all_results).to_json(OUTPUT_FILE, orient="records", indent=4)
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user.")
            break
        except Exception as e:
            pbar.write(f"✘ Error on {city_name}: {e}")

    if all_results:
        print(f"\nSUCCESS: Data saved to {OUTPUT_FILE}")
    else:
        print("\nNo data collected.")

if __name__ == "__main__":
    main()