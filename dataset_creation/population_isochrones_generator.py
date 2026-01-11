#!/usr/bin/env python3
"""
ISO4App In-Place Enrichment
Updates final_clustered_samples.json directly, skipping already enriched points
"""
import json
import time
import requests
import os
import shutil
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FILE = "./data/final_clustered_samples.json"
OUTPUT_FILE = "./data/final_clustered_samples.json" 
BACKUP_FILE = "./data/final_clustered_samples_backup.json"  

# ISO4App Configuration
ISOLINE_BASE_URL = "http://www.iso4app.net/rest/1.3/isoline.geojson"
POPULATION_BASE_URL = "https://api.iso4app.com"
LICENSE_KEY = "YOUR_API_KEY"

# Walking parameters
WALKING_TIME_MINUTES = 15
WALKING_TIME_SECONDS = WALKING_TIME_MINUTES * 60
MOBILITY_TYPE = "pedestrian"

# Credentials - UPDATE THESE!
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"

# API settings
DELAY_BETWEEN_CALLS = 5  # seconds between points
SAVE_INTERVAL = 50  # Save every N points
TIMEOUT = 30


# === API FUNCTIONS ===
def get_walking_isochrone(lat: float, lon: float) -> Tuple[Optional[Dict], Optional[float], Optional[Dict]]:
    """
    Get walking isochrone using public API.
    Returns: (geometry, approximation_meters, start_point) or (None, None, None)
    """
    params = {
        "licKey": LICENSE_KEY,
        "type": "isochrone",  
        "value": WALKING_TIME_SECONDS,
        "lat": lat,
        "lng": lon,
        "mobility": MOBILITY_TYPE
    }
    
    try:
        response = requests.get(ISOLINE_BASE_URL, params=params, timeout=TIMEOUT)
        
        if response.ok:
            geojson = response.json()
            
            geometry = None
            approximation = None
            start_point = None
            
            for feature in geojson.get("features", []):
                props = feature.get("properties", {})
                
                if props.get("name") == "isoline":
                    geometry = feature.get("geometry")
                
                if props.get("name") == "start point":
                    approximation = props.get("approximation")
                    coords = feature.get("geometry", {}).get("coordinates")
                    if coords:
                        start_point = {"lat": coords[1], "lng": coords[0]}
            
            return geometry, approximation, start_point
        else:
            return None, None, None
            
    except Exception:
        return None, None, None


def polygon_to_iso4app_format(geometry: Dict) -> Optional[List[Dict]]:
    """Convert GeoJSON polygon to ISO4App format"""
    try:
        if geometry["type"] == "Polygon":
            coords = geometry["coordinates"][0]
        elif geometry["type"] == "MultiPolygon":
            coords = geometry["coordinates"][0][0]
        else:
            return None
        
        return [{"lat": point[1], "lng": point[0]} for point in coords]
    except Exception:
        return None


def login_iso4app(username: str, password: str) -> Optional[str]:
    """Login and return access token"""
    try:
        response = requests.post(
            f"{POPULATION_BASE_URL}/user/login",
            json={
                "username": username,
                "password": password,
                "clientVersion": "1.0.0"
            },
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=TIMEOUT
        )
        
        if response.ok:
            return response.json()["accessToken"]
        else:
            return None
    except Exception:
        return None


def get_population_for_polygon(auth_token: str, polygon_coords: List[Dict]) -> Optional[float]:
    """Get population for a polygon"""
    try:
        txt_coords = ",".join([f"{p['lat']} {p['lng']}" for p in polygon_coords])
        
        response = requests.post(
            f"{POPULATION_BASE_URL}/indicator",
            json={
                "polygon": txt_coords,
                "category": 1000,
                "returnOnlyValue": True
            },
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {auth_token}"
            },
            timeout=TIMEOUT
        )
        
        if response.ok:
            return response.json().get("v", 0)
        else:
            return None
    except Exception:
        return None


def is_point_already_enriched(point: Dict) -> bool:
    """Check if a point already has population data"""
    return 'population_15min_walk' in point and point['population_15min_walk'] is not None


def save_data(data: list, filepath: str):
    """Save data to file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


# === MAIN ENRICHMENT ===
def enrich_samples():
    """Main enrichment function - updates file in place"""
    print("\n" + "="*70)
    print("ISO4APP IN-PLACE ENRICHMENT")
    print("="*70)
    print(f"File: {INPUT_FILE}")
    print(f"Walking time: {WALKING_TIME_MINUTES} minutes (ISOCHRONOUS)")
    print(f"Updates existing file - skips already enriched points")
    print("="*70 + "\n")
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Load input data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} points from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_FILE}")
        return
    
    # Create backup (only if it doesn't exist yet)
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        print(f"✓ Created backup: {BACKUP_FILE}")
    else:
        print(f"✓ Backup already exists: {BACKUP_FILE}")
    
    # Check which points need enrichment
    already_enriched = [i for i, p in enumerate(data) if is_point_already_enriched(p)]
    needs_enrichment = [i for i, p in enumerate(data) if not is_point_already_enriched(p)]
    
    print(f"\n📊 Status:")
    print(f"   Total points: {len(data)}")
    print(f"   ✓ Already enriched: {len(already_enriched)}")
    print(f"   🚀 Need enrichment: {len(needs_enrichment)}")
    
    if not needs_enrichment:
        print("\n✓ All points already enriched! Nothing to do.")
        return
    
    print(f"   ⏱️  Estimated time: ~{(len(needs_enrichment) * DELAY_BETWEEN_CALLS) / 60:.1f} minutes")
    print()
    
    # Login
    print("🔐 Logging in to ISO4App...")
    auth_token = login_iso4app(USERNAME, PASSWORD)
    
    if not auth_token:
        print("❌ Login failed! Check credentials.")
        print(f"   USERNAME: {USERNAME}")
        return
    
    print("✓ Login successful\n")
    
    # Process points
    successful = 0
    failed = 0
    
    pbar = tqdm(needs_enrichment, desc="Enriching", unit="point")
    
    for idx in pbar:
        point = data[idx]
        
        try:
            city = point.get('city', 'Unknown').split(',')[0]
            pbar.set_description(f"{city}")
            
            # Step 1: Get isochrone
            geometry, approximation, start_point = get_walking_isochrone(
                point['lat'], point['lon']
            )
            
            if geometry is None:
                point['walking_isochrone'] = None
                point['population_15min_walk'] = None
                point['approximation_meters'] = None
                point['start_point'] = None
                point['error'] = "Failed to get isochrone"
                failed += 1
                time.sleep(DELAY_BETWEEN_CALLS)  # 5 seconds before next point
                continue
            
            # Convert polygon
            polygon_coords = polygon_to_iso4app_format(geometry)
            
            if polygon_coords is None:
                point['walking_isochrone'] = None
                point['population_15min_walk'] = None
                point['approximation_meters'] = None
                point['start_point'] = None
                point['error'] = "Failed to convert polygon"
                failed += 1
                time.sleep(DELAY_BETWEEN_CALLS)  # 5 seconds before next point
                continue
            
            # Store isochrone data
            point['walking_isochrone'] = geometry
            point['approximation_meters'] = approximation
            point['start_point'] = start_point
            
            time.sleep(0.5)
            
            # Step 2: Get population
            population = get_population_for_polygon(auth_token, polygon_coords)
            
            if population is None:
                point['population_15min_walk'] = None
                point['error'] = "Failed to get population"
                failed += 1
            else:
                point['population_15min_walk'] = population
                successful += 1
                # Remove error field if it existed
                if 'error' in point:
                    del point['error']
            
            # Save progress periodically
            if (successful + failed) % SAVE_INTERVAL == 0:
                save_data(data, OUTPUT_FILE)
                pbar.write(f"💾 Progress saved ({successful} successful, {failed} failed)")
            
            # Wait 5 seconds total before next point (already waited 0.5s above)
            time.sleep(DELAY_BETWEEN_CALLS - 0.5)
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted. Saving progress...")
            save_data(data, OUTPUT_FILE)
            print(f"✓ Saved. {successful} enriched, {failed} failed.")
            print(f"   Run again to continue from point {idx + 1}")
            return
        
        except Exception as e:
            pbar.write(f"❌ Error at point {idx}: {e}")
            point['walking_isochrone'] = None
            point['population_15min_walk'] = None
            point['approximation_meters'] = None
            point['start_point'] = None
            point['error'] = str(e)
            failed += 1
    
    # Final save
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"✅ Successfully enriched: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"💾 Saving final results...")
    
    save_data(data, OUTPUT_FILE)
    
    print(f"✓ File updated: {OUTPUT_FILE}")
    print(f"✓ Backup available: {BACKUP_FILE}")
    
    # Statistics
    if successful > 0:
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        
        populations = [p.get('population_15min_walk') for p in data 
                      if p.get('population_15min_walk') is not None]
        
        if populations:
            import statistics
            print(f"Total enriched points: {len(populations)}")
            print(f"Avg population: {statistics.mean(populations):,.0f}")
            print(f"Median: {statistics.median(populations):,.0f}")
            print(f"Max: {max(populations):,.0f}")
            print(f"Min: {min(populations):,.0f}")
        
        approximations = [p.get('approximation_meters') for p in data 
                         if p.get('approximation_meters') is not None]
        
        if approximations:
            import statistics
            print(f"\nApproximation distances:")
            print(f"Avg distance to street: {statistics.mean(approximations):.1f}m")
            print(f"Median: {statistics.median(approximations):.1f}m")
            print(f"Max: {max(approximations):.1f}m")
            print(f"Min: {min(approximations):.1f}m")
            
            far = len([a for a in approximations if a > 100])
            very_far = len([a for a in approximations if a > 250])
            print(f"Points >100m from street: {far} ({far/len(approximations)*100:.1f}%)")
            print(f"Points >250m from street: {very_far} ({very_far/len(approximations)*100:.1f}%)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    enrich_samples()
