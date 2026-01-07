# pip install rasterio eodag geopandas matplotlib
import os
import rasterio
from rasterio.windows import Window
import numpy as np

def split_image_into_patches(image_path, output_folder, patch_size=256):
    """
    Splits a large satellite image into fixed-size patches.
    
    Args:
        image_path (str): Path to the .jp2 or .tif file (e.g., the TCI_10m band).
        output_folder (str): Where to save the chips.
        patch_size (int): Size of the square patch (e.g., 256 pixels).
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    with rasterio.open(image_path) as src:
        # Get image dimensions
        width = src.width
        height = src.height
        
        print(f"Processing image: {width}x{height} pixels")
        
        # Loop through the image with a sliding window
        patch_id = 0
        
        # STEP: simple non-overlapping grid
        for col_off in range(0, width, patch_size):
            for row_off in range(0, height, patch_size):
                
                # Calculate window position
                # We must handle edges (if city size isn't a multiple of 256)
                # Option A: Skip partial patches (cleanest for training)
                if col_off + patch_size > width or row_off + patch_size > height:
                    continue
                    
                # Define the window to read
                window = Window(col_off, row_off, patch_size, patch_size)
                
                # Read the data in that window
                # (1, 2, 3) means read Red, Green, Blue bands
                patch_data = src.read((1, 2, 3), window=window)
                
                # OPTIONAL: Filter out empty/black patches (common in satellite data)
                if np.mean(patch_data) == 0:
                    continue

                # Save the patch
                # We need to update the metadata (transform) for the small patch
                patch_transform = src.window_transform(window)
                patch_meta = src.meta.copy()
                patch_meta.update({
                    "driver": "GTiff",
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "count": 3
                })
                
                out_filename = os.path.join(output_folder, f"patch_{patch_id}.tif")
                with rasterio.open(out_filename, "w", **patch_meta) as dest:
                    dest.write(patch_data)
                
                patch_id += 1
                
        print(f"Created {patch_id} patches for this city.")

# --- Usage Example ---
# 1. Download a Sentinel-2 product (zip) manually or via EODAG
# 2. Extract the zip and find the "TCI" (True Color Image) file ending in .jp2
#    Example path usually looks like: .../GRANULE/L2A_.../IMG_DATA/R10m/TCI_...jp2

# split_image_into_patches("path/to/TCI_image.jp2", "dataset/city_name/patches")