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

# Target years for out-of-sample projection
TARGET_YEARS = [2025, 2026]

# Define Florida Boundary
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))

# =====================================================================
# 2. GENERATE THE IDENTICAL 5KM POINT GRID
# =====================================================================
print("Generating native 5km point grid...")
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
# 3. DEFINE DATA COLLECTIONS & EXTRACTION MATH
# =====================================================================

# --- Daymet V4 Daily Weather ---
daymet_col = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
    .filterBounds(florida.geometry()) \
    .select(['tmax', 'tmin', 'prcp', 'vp'])

def extract_daymet(image):
    date = image.date().format('YYYY-MM-dd')
    tmax = image.select('tmax')
    tmin = image.select('tmin')
    tmean = tmax.add(tmin).divide(2).rename('tmean')
    
    # Tetens Formula for Saturation Vapor Pressure (es) and VPD
    t_exp = tmean.multiply(17.27).divide(tmean.add(237.3)).exp()
    es = t_exp.multiply(610.78).rename('es')
    vpd = es.subtract(image.select('vp')).rename('vpd')
    
    final_image = image.addBands([tmean, vpd])
    extracted = final_image.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False
    )
    return extracted.map(lambda feature: feature.set('Date', date))

# --- MODIS EVI (16-day) ---
evi_col = ee.ImageCollection("MODIS/061/MOD13A1") \
    .filterBounds(florida.geometry()) \
    .select(['EVI'])

def extract_evi(image):
    date = image.date().format('YYYY-MM-dd')
    scaled_evi = image.select('EVI').multiply(0.0001).rename('EVI')
    extracted = scaled_evi.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False
    )
    return extracted.map(lambda feature: feature.set('Date', date))

# --- MODIS NDWI via Surface Reflectance (8-day) ---
refl_col = ee.ImageCollection("MODIS/061/MOD09A1") \
    .filterBounds(florida.geometry()) \
    .select(['sur_refl_b04', 'sur_refl_b02']) # Band 4 = Green, Band 2 = NIR

def extract_ndwi(image):
    date = image.date().format('YYYY-MM-dd')
    ndwi = image.normalizedDifference(['sur_refl_b04', 'sur_refl_b02']).rename('NDWI')
    extracted = ndwi.sampleRegions(
        collection=grid_points,
        scale=config['scale_meters'],
        projection=metric_proj,
        geometries=False
    )
    return extracted.map(lambda feature: feature.set('Date', date))

# =====================================================================
# 4. BATCH ORCHESTRATION LOOP
# =====================================================================
print(f"\nSending batch exports to Google Drive for years: {TARGET_YEARS}...")

for year in TARGET_YEARS:
    yearly_start = f"{year}-01-01"
    yearly_end = f"{year}-12-31"
    
    # 1. Daymet Export
    yearly_daymet = daymet_col.filterDate(yearly_start, yearly_end)
    task_daymet = ee.batch.Export.table.toDrive(
        collection=yearly_daymet.map(extract_daymet).flatten(),
        description=f"FL_Daymet_5km_{year}",
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'tmax', 'tmin', 'tmean', 'prcp', 'vpd']
    )
    task_daymet.start()
    
    # 2. MODIS EVI Export
    yearly_evi = evi_col.filterDate(yearly_start, yearly_end)
    task_evi = ee.batch.Export.table.toDrive(
        collection=yearly_evi.map(extract_evi).flatten(),
        description=f"FL_EVI_5km_{year}",
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'EVI']
    )
    task_evi.start()
    
    # 3. MODIS NDWI Export
    yearly_ndwi = refl_col.filterDate(yearly_start, yearly_end)
    task_ndwi = ee.batch.Export.table.toDrive(
        collection=yearly_ndwi.map(extract_ndwi).flatten(),
        description=f"FL_NDWI_5km_{year}",
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'NDWI']
    )
    task_ndwi.start()
    
    print(f"[{year}] Queued tasks: Daymet, EVI, and NDWI.")

print("\nAll tasks submitted! Check your Earth Engine Code Editor 'Tasks' tab.")