import ee
import json
import geemap
import os

# 1. Load your configuration
print("Loading config.json...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# 2. Initialize Earth Engine 
ee.Initialize(project=config['gcp_project_id']) 
print("Connected to Earth Engine.")

# 3. Define the boundary
states = ee.FeatureCollection("TIGER/2018/States")
target_geometry = states.filter(ee.Filter.eq('NAME', config['target_state'])).first().geometry()

# 4. Grab the data and crush it into a single multi-band layer
print(f"Fetching Daymet data for {config['target_state']}...")
collection = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
    .filterBounds(target_geometry) \
    .filterDate(config['start_date'], config['end_date']) \
    .select(['tmax', 'tmin'])

layered_image = collection.toBands().clip(target_geometry)

# 5. Set up the local file path
dynamic_filename = f"{config['file_prefix']}_{config['start_date']}_to_{config['end_date']}_{config['scale_meters']}m.tif"

# os.getcwd() gets your "Current Working Directory" (your VS Code folder)
# This joins your folder path with the new filename
out_tif_path = os.path.join(os.getcwd(), dynamic_filename)

# 6. Download DIRECTLY to your computer
print(f"Downloading {dynamic_filename} directly to your VS Code folder...")
print("(This might take a minute depending on your internet connection...)")

try:
    geemap.ee_export_image(
        layered_image,
        filename=out_tif_path,
        scale=config['scale_meters'],
        region=target_geometry,
        file_per_band=False # Keeps all 60 layers safely inside one single .tif file
    )
    print(f"\nSuccess! File downloaded directly to: {out_tif_path}")
except Exception as e:
    print(f"\nDownload Failed. The file might be too large for a direct download (>32MB).")
    print(f"Error details: {e}")