#!/usr/bin/env bash
# =============================================================================
#  Download OptimalAccessDataset.zip from Google Drive and extract it.
#
#  Usage:
#    bash download_dataset.sh <GOOGLE_DRIVE_FILE_ID>
#
#  The file ID is the part between /d/ and /view in the share URL:
#    https://drive.google.com/file/d/THIS_PART_HERE/view?usp=sharing
#
#  Requirements:
#    - File must be shared as "Anyone with the link" in Google Drive
# =============================================================================
set -e

FILE_ID="${1}"
DEST_DIR="/workspace/PopulationDataset"
ZIP_PATH="/workspace/OptimalAccessDataset.zip"

if [ -z "$FILE_ID" ]; then
    echo "Usage: bash download_dataset.sh <GOOGLE_DRIVE_FILE_ID>"
    echo ""
    echo "Example:"
    echo "  bash download_dataset.sh 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
    echo ""
    echo "Find your file ID in the Google Drive share link:"
    echo "  https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing"
    exit 1
fi

echo "===== 1. Installing gdown ====="
pip install -q gdown

echo ""
echo "===== 2. Downloading from Google Drive ====="
echo "  File ID : $FILE_ID"
echo "  Saving to: $ZIP_PATH"
gdown --fuzzy "https://drive.google.com/file/d/${FILE_ID}/view" -O "$ZIP_PATH"

echo ""
echo "===== 3. Extracting ====="
mkdir -p "$DEST_DIR"
unzip -q "$ZIP_PATH" -d "$DEST_DIR"
echo "  Extracted to $DEST_DIR"

echo ""
echo "===== 4. Verifying structure ====="
for d in satellite_images dem_height segmentation_land_use; do
    if [ -d "$DEST_DIR/$d" ]; then
        count=$(ls "$DEST_DIR/$d" | wc -l)
        echo "  ✓ $d  ($count files)"
    else
        echo "  ✗ MISSING: $d — check the zip structure"
    fi
done

if [ -f "$DEST_DIR/final_clustered_samples.json" ]; then
    echo "  ✓ final_clustered_samples.json"
else
    echo "  ✗ MISSING: final_clustered_samples.json"
fi

echo ""
echo "===== 5. Cleaning up zip ====="
rm "$ZIP_PATH"
echo "  Deleted $ZIP_PATH"

echo ""
echo "Done! Dataset ready at $DEST_DIR"
