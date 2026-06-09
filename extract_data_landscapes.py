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

states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))

# =====================================================================
# 2. GENERATE THE PERFECT MATCHING GRID
# =====================================================================
print("Generating identical 5km grid infrastructure...")
metric_proj = ee.Projection('EPSG:3857').atScale(config['scale_meters'])

# Step A: Generate the exact same points
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

# Step B: THE PERCENTAGE FIX (Create 5km boxes around the points)
# We buffer the point by 2500m and take the bounds to create a 5km x 5km square.
# Crucially, this square permanently inherits the Grid_ID of the center point.
grid_boxes = grid_points.map(lambda f: f.buffer(2500).bounds())

print("Grid generation complete.")

# =====================================================================
# 3. EXTRACT TOPOGRAPHY (RUNS ONCE)
# =====================================================================
print("\nQueueing Topography (Elevation & Slope)...")

srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
slope = ee.Terrain.slope(srtm).rename('slope')
topo_image = srtm.addBands(slope)

# Calculate the average elevation and slope inside the 5km box
tabular_topo = topo_image.reduceRegions(
    collection=grid_boxes,
    reducer=ee.Reducer.mean(),
    scale=30 # SRTM native resolution
)

# Export Topography
task_topo = ee.batch.Export.table.toDrive(
    collection=tabular_topo,
    description="FL_Topography_5km",
    folder='Masters_Project_Data',
    fileFormat='CSV',
    selectors=['Grid_ID', 'elevation', 'slope']
)
task_topo.start()
print("Topography task successfully sent to GEE.")

# =====================================================================
# 4. EXTRACT LAND USE (LOOPED PER YEAR)
# =====================================================================
start_year = config['start_year']
end_year = config['end_year']

print(f"\nQueueing Land Cover exports from {start_year} to {end_year}...")

# THE FIX: We no longer load the entire NLCD ImageCollection. 
# We will construct the exact paths inside the loop.

for year in range(start_year, end_year + 1):
    
    # Match the study year to the closest NLCD satellite release
    if year <= 2014:
        nlcd_year = '2013'
        release_folder = '2019_REL'
    elif year <= 2017:
        nlcd_year = '2016'
        release_folder = '2019_REL'
    else:
        nlcd_year = '2019'
        release_folder = '2019_REL' # 2019 also lives in the 2019 release folde
        
    print(f"[{year}] Pairing with NLCD {nlcd_year} database...")
    
    print(f"[{year}] Pairing with NLCD {nlcd_year} database...")
    
    # THE FIX: Dynamically route to the correct Release Folder
    exact_image_path = f"USGS/NLCD_RELEASES/{release_folder}/NLCD/{nlcd_year}"
    
    # Load the specific image directly and select the band, bypassing filters
    landcover = ee.Image(exact_image_path).select('landcover')
    
    # Count every single pixel type inside the 5km box
    tabular_landcover = landcover.reduceRegions(
        collection=grid_boxes,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30 # NLCD native resolution
    )
    
    # Add the Date so it merges cleanly with your weather data later
    date_string = f"{year}-01-01"
    tabular_landcover = tabular_landcover.map(lambda feature: feature.set('Date', date_string))
    
    task_name = f"FL_LandCover_5km_{year}"
    
    task_land = ee.batch.Export.table.toDrive(
        collection=tabular_landcover,
        description=task_name,
        folder='Masters_Project_Data',
        fileFormat='CSV'
    )
    task_land.start()

print("\nSuccess! All Static and Land Use tasks are safely queued.")