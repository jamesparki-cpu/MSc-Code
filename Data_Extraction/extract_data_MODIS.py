import ee
import json

# =====================================================================
# 1. INITIALIZATION & CONFIGURATION
# =====================================================================
print("Loading config.json...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

ee.Initialize(project=config['gcp_project_id'])
print("Connected to Earth Engine.")

# Define Florida Boundary
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))

# =====================================================================
# 2. GENERATE THE EXACT SAME 5KM POINT GRID (CRITICAL FOR MERGING)
# =====================================================================
print("Generating native 5km point grid to match Daymet...")

metric_proj = ee.Projection('EPSG:3857').atScale(config['scale_meters'])

grid_points = ee.Image.pixelLonLat().mask(ee.Image().paint(florida, 1)).sample(
    region=florida.geometry(),
    scale=config['scale_meters'],
    projection=metric_proj,
    geometries=True, 
    dropNulls=True
)

def assign_grid_id(feature):
    lat = ee.Number(feature.get('latitude')).format('%.3f')
    lon = ee.Number(feature.get('longitude')).format('%.3f')
    grid_id = ee.String('Grid_').cat(lat).cat('_').cat(lon)
    return feature.set('Grid_ID', grid_id)

grid_points = grid_points.map(assign_grid_id)
print("Grid generation complete.")

# =====================================================================
# 3. DEFINE MODIS SATELLITE DATA
# =====================================================================

# EVI is captured every 16 days
evi_collection = ee.ImageCollection("MODIS/061/MOD13A1") \
    .filterBounds(florida.geometry()) \
    .select(['EVI'])

# Surface Reflectance (used for NDWI) is captured every 8 days
reflectance_collection = ee.ImageCollection("MODIS/061/MOD09A1") \
    .filterBounds(florida.geometry()) \
    .select(['sur_refl_b04', 'sur_refl_b02']) # Band 4 = Green, Band 2 = NIR

# =====================================================================
# 4. POINT EXTRACTION FUNCTIONS
# =====================================================================

def extract_evi(image):
    date = image.date().format('YYYY-MM-dd')
    
    # Scale EVI back to true biological bounds (-1 to 1)
    scaled_evi = image.multiply(0.0001).rename('EVI')
    
    extracted = scaled_evi.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False
    )
    return extracted.map(lambda feature: feature.set('Date', date))

def extract_ndwi(image):
    date = image.date().format('YYYY-MM-dd')
    
    # NDWI formula: (Green - NIR) / (Green + NIR)
    ndwi = image.normalizedDifference(['sur_refl_b04', 'sur_refl_b02']).rename('NDWI')
    
    extracted = ndwi.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False
    )
    return extracted.map(lambda feature: feature.set('Date', date))

# =====================================================================
# 5. BATCH ORCHESTRATION LOOP
# =====================================================================

start_year = config['start_year']
end_year = config['end_year']

print(f"\nSending MODIS batch exports to Google Drive from {start_year} to {end_year}...")

for year in range(start_year, end_year + 1):
    yearly_start = f"{year}-01-01"
    yearly_end = f"{year}-12-31"
    
    # --- Process EVI (16-day cycle) ---
    yearly_evi = evi_collection.filterDate(yearly_start, yearly_end)
    tabular_evi = yearly_evi.map(extract_evi).flatten()
    
    task_evi = ee.batch.Export.table.toDrive(
        collection=tabular_evi,
        description=f"FL_EVI_5km_{year}",
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'EVI']
    )
    task_evi.start()
    
    # --- Process NDWI (8-day cycle) ---
    yearly_ndwi = reflectance_collection.filterDate(yearly_start, yearly_end)
    tabular_ndwi = yearly_ndwi.map(extract_ndwi).flatten()
    
    task_ndwi = ee.batch.Export.table.toDrive(
        collection=tabular_ndwi,
        description=f"FL_NDWI_5km_{year}",
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'NDWI']
    )
    task_ndwi.start()
    
    print(f"[{year}] EVI and NDWI tasks successfully queued.")

print("\nSuccess! All MODIS tasks are queued. Check the Earth Engine Code Editor 'Tasks' tab.")