import json
import os
import rioxarray
import numpy as np
import xarray as xr
from pystac_client import Client
import odc.stac

# --- CONFIGURATION ---
INPUT_FILE = "./data/cities_bboxes_major_europe.json"
OUTPUT_DIR = "./imgs/satellite_images"

# 🟢 CHANGED: We now accept practically ANY image (up to 95% clouds).
# We do this to ensure we catch the 'edges' of the satellite path.
# We will filter out the actual cloud pixels later using the SCL band.
MAX_CLOUD_COVER = 95 

COUNTRY_DATE_MAP = {
    "Italy": "2023-06-01/2023-09-01", 
    "Denmark": "2023-06-01/2023-09-01",
    "Norway": "2023-06-01/2023-09-01",
    "Finland": "2022-06-01/2022-09-01",
    "Netherlands": "2022-06-01/2022-09-01",
    "Austria": "2020-06-01/2020-09-01",
    "Belgium": "2020-06-01/2020-09-01",
    "France": "2020-06-01/2020-09-01",
    "Greece": "2020-06-01/2020-09-01",
    "Sweden": "2020-06-01/2020-09-01",
    "United Kingdom": "2020-06-01/2020-09-01",
    "Portugal": "2018-06-01/2018-09-01",
    "Switzerland": "2018-06-01/2018-09-01"
}
DEFAULT_TIME_RANGE = "2023-06-01/2023-09-01"

def download_satellite_image(city_name, bbox_coords, time_range):
    print(f"\n🛰 Processing {city_name}...")
    
    lats = [p[0] for p in bbox_coords]
    lons = [p[1] for p in bbox_coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    
    items = list(search.item_collection())
    print(f"   🔍 Found {len(items)} candidate tiles (Limit < {MAX_CLOUD_COVER}% clouds)")
    
    if len(items) == 0:
        print("   ❌ No images found at all.")
        return

    try:
        # 🟢 CHANGED: Load 'scl' (Scene Classification Layer) along with RGB
        # SCL tells us which pixels are clouds, water, or land.
        data = odc.stac.load(
            items,
            bands=["red", "green", "blue", "scl"], 
            bbox=bbox,
            crs="EPSG:3857", 
            resolution=10,
            groupby="solar_day",
            stac_cfg={"sentinel-2-l2a": {"assets": {"*": {"data_type": "uint16", "nodata": 0}}}}
        )
        
        # --- PIXEL MASKING MAGIC ---
        # SCL Classes:
        # 0=No Data, 1=Saturated, 3=Cloud Shadow, 8=Medium Cloud, 9=High Cloud, 10=Cirrus
        # We want to KEEP: 4 (Vegetation), 5 (Bare Soil), 6 (Water), 7 (Unclassified)
        
        # 1. Create a "Good Pixel" mask (True = Good, False = Bad)
        # We explicitly exclude the bad classes (clouds, shadows, nodata)
        bad_pixels = data.scl.isin([0, 1, 3, 8, 9, 10, 11])
        
        # 2. Apply mask to RGB bands (turn bad pixels into NaN)
        # This effectively "erases" the clouds from every single image layer
        clean_data = data[["red", "green", "blue"]].where(~bad_pixels)
        
        # 3. Calculate Median
        # xarray's median ignores NaNs automatically. 
        # So it looks through the stack of images and picks the middle value 
        # from the valid (non-cloudy) pixels only.
        image = clean_data.median(dim="time").to_array("band")
        
        # 4. Fill remaining gaps (if any persistent holes exist) with black (0)
        image = image.fillna(0)

        # Re-attach CRS
        image.rio.write_crs("EPSG:3857", inplace=True)
        
        # Normalize brightness & Clip
        image = image.clip(0, 3000) / 3000 * 255
        image = image.astype("uint8")
        
        safe_name = city_name.replace(", ", "_").replace(" ", "_")
        filename = f"{OUTPUT_DIR}/{safe_name}_10m.tif"
        
        image.rio.to_raster(filename)
        print(f"   ✔ Saved to {filename}")
        
    except Exception as e:
        print(f"   ✘ Failed to process {city_name}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
            cities_list = data.get("cities", [])
            
        print(f"📂 Loaded {len(cities_list)} cities.")
        
        for entry in cities_list:
            city_str = entry.get('city')
            bbox = entry.get('bbox')
            
            safe_name = city_str.replace(", ", "_").replace(" ", "_")
            if os.path.exists(f"{OUTPUT_DIR}/{safe_name}_10m.tif"):
                print(f"⏩ Skipping {city_str} (Already exists)")
                continue
            
            try:
                country = city_str.split(", ")[-1].strip()
                target_date = COUNTRY_DATE_MAP.get(country, DEFAULT_TIME_RANGE)
            except:
                target_date = DEFAULT_TIME_RANGE

            download_satellite_image(city_str, bbox, target_date)
                
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_FILE}")

if __name__ == "__main__":
    main()