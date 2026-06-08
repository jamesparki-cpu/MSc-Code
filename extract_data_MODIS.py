import ee
import json
import geemap

# 1. Load Configuration & Initialize
print("Loading config.json...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

ee.Initialize(project=config['gcp_project_id'])
print("Connected to Earth Engine.")

# 2. Setup Spatial Boundary and Grid
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))
grid_5km = geemap.fishnet(
    florida.geometry(), 
    h_interval=config['scale_meters'], 
    v_interval=config['scale_meters'], 
    keys=['Grid_ID']
)

# 3. Define the Main Collections
# MOD13A1 provides EVI directly at 500m resolution every 16 days.
evi_collection = ee.ImageCollection("MODIS/061/MOD13A1") \
    .filterBounds(florida.geometry()) \
    .select(['EVI'])

# MOD09A1 provides surface reflectance every 8 days. 
# We use this to calculate NDWI.
reflectance_collection = ee.ImageCollection("MODIS/061/MOD09A1") \
    .filterBounds(florida.geometry()) \
    .select(['sur_refl_b04', 'sur_refl_b02']) # Band 4 = Green, Band 2 = NIR

# 4. Extraction Functions
def extract_evi(image):
    date = image.date().format('YYYY-MM-dd')
    # MODIS EVI requires a scale factor of 0.0001
    scaled_evi = image.multiply(0.0001).rename('EVI')
    
    reduced = scaled_evi.reduceRegions(
        collection=grid_5km,
        reducer=ee.Reducer.mean(),
        scale=500 # Native MODIS resolution
    )
    return reduced.map(lambda feature: feature.set('Date', date))

def extract_ndwi(image):
    date = image.date().format('YYYY-MM-dd')
    # Calculate NDWI = (Green - NIR) / (Green + NIR)
    ndwi = image.normalizedDifference(['sur_refl_b04', 'sur_refl_b02']).rename('NDWI')
    
    reduced = ndwi.reduceRegions(
        collection=grid_5km,
        reducer=ee.Reducer.mean(),
        scale=500
    )
    return reduced.map(lambda feature: feature.set('Date', date))

# 5. Batch Orchestration Loop
start_year = config['study_start_year']
end_year = config['study_end_year']

print(f"\nOrchestrating MODIS exports from {start_year} to {end_year}...")

for year in range(start_year, end_year + 1):
    yearly_start = f"{year}-01-01"
    yearly_end = f"{year}-12-31"
    
    # Process EVI (16-day)
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
    
    # Process NDWI (8-day)
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
    
    print(f"[{year}] EVI and NDWI tasks successfully sent to GEE.")

print("\nAll MODIS tasks submitted! Check the GEE Tasks tab.")