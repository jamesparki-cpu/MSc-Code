import ee
import json

# 1. Initialize
print("Loading config.json...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

ee.Initialize(project=config['gcp_project_id'])
print("Connected to Earth Engine.")

# 2. Define Florida Boundary
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))

# =====================================================================
# THE FIX: Create a mathematically perfect 5km Point Grid natively in GEE
# =====================================================================
print("Generating native 5km point grid...")

# Define a metric projection (EPSG:3857 is Web Mercator) at exactly 5000m scale
metric_proj = ee.Projection('EPSG:3857').atScale(config['scale_meters'])

# Create an image of coordinates, mask it to Florida, and convert it to points
grid_points = ee.Image.pixelLonLat().mask(ee.Image().paint(florida, 1)).sample(
    region=florida.geometry(),
    scale=config['scale_meters'],
    projection=metric_proj,
    geometries=True, # Keeps the GPS coordinates for the CSV
    dropNulls=True
)

# Assign a permanent, unique Grid_ID to each point based on its coordinates
def assign_grid_id(feature):
    # Extracts Lat/Lon and formats them as a string (e.g., "Grid_28.5_-81.2")
    lat = ee.Number(feature.get('latitude')).format('%.3f')
    lon = ee.Number(feature.get('longitude')).format('%.3f')
    grid_id = ee.String('Grid_').cat(lat).cat('_').cat(lon)
    return feature.set('Grid_ID', grid_id)

grid_points = grid_points.map(assign_grid_id)
print("Grid generation complete.")

# =====================================================================
# MAIN CLIMATE MATH & EXTRACTION
# =====================================================================

main_climate_collection = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
    .filterBounds(florida.geometry()) \
    .select(['tmax', 'tmin', 'prcp', 'vp'])

def process_and_extract(image):
    date = image.date().format('YYYY-MM-dd')
    
    # 1. Calculate Mean Temp (tmean)
    tmax = image.select('tmax')
    tmin = image.select('tmin')
    tmean = tmax.add(tmin).divide(2).rename('tmean')
    
    # 2. Calculate Saturation Vapor Pressure (es) for VPD
    # Tetens Formula
    t_exp = tmean.multiply(17.27).divide(tmean.add(237.3)).exp()
    es = t_exp.multiply(610.78).rename('es')
    
    # 3. Calculate VPD (es - actual vapor pressure)
    vpd = es.subtract(image.select('vp')).rename('vpd')
    
    # Stack all bands together
    final_image = image.addBands([tmean, vpd])
    
    # 4. Extract values perfectly at the 5km point grid
    # sampleRegions is highly optimized for point extraction
    extracted = final_image.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False # Set to False to keep CSV clean (Lat/Lon are already stored in Grid_ID)
    )
    
    # Attach the date to every single row
    return extracted.map(lambda feature: feature.set('Date', date))


# =====================================================================
# BATCH ORCHESTRATION LOOP
# =====================================================================

start_year = config['start_year']
end_year = config['end_year']

print(f"\nSending batch exports to Google Drive from {start_year} to {end_year}...")

for year in range(start_year, end_year + 1):
    yearly_start = f"{year}-01-01"
    yearly_end = f"{year}-12-31"
    
    yearly_collection = main_climate_collection.filterDate(yearly_start, yearly_end)
    
    # Apply the math and extraction function
    tabular_data = yearly_collection.map(process_and_extract).flatten()
    
    task_name = f"FL_Daymet_5km_{year}"
    
    export_task = ee.batch.Export.table.toDrive(
        collection=tabular_data,
        description=task_name,
        folder='Masters_Project_Data',
        fileFormat='CSV',
        # Explicitly declare every column you want so none are dropped
        selectors=['Grid_ID', 'Date', 'tmax', 'tmin', 'tmean', 'prcp', 'vpd']
    )
    
    export_task.start()
    print(f"[{year}] Sent to GEE Servers.")

print("\nSuccess! All tasks are queued. Check the Earth Engine Code Editor 'Tasks' tab.")