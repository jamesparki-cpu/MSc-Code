import ee
import json
import geemap

# 1. Load Configuration
print("Loading config.json...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize Earth Engine
ee.Initialize(project=config['gcp_project_id'])
print("Connected to Earth Engine.")

# 2. Setup Spatial Boundary and Grid
print(f"Generating 5km grid for {config['target_state']}...")
states = ee.FeatureCollection("TIGER/2018/States")
florida = states.filter(ee.Filter.eq('NAME', config['target_state']))

# Create the 5km fishnet grid
grid_5km = geemap.fishnet(
    florida.geometry(), 
    h_interval=config['scale_meters'], 
    v_interval=config['scale_meters'], 
    keys=['Grid_ID']
)

# 3. Define the Main Climate Collection (FIX 1)
# This was missing in the previous snippet. We define the massive base dataset here 
# before the loop so we can filter it down year-by-year inside the loop.
main_climate_collection = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
    .filterBounds(florida.geometry()) \
    .select(['tmax', 'tmin', 'prcp', 'vp'])

# 4. Define the Extraction Function
# This tells GEE how to turn a raster image into a tabular DataFrame
def extract_tabular_data(image):
    date = image.date().format('YYYY-MM-dd')

    # 1. Calculate Mean Temperature
    tmean = image.select('tmax').add(image.select('tmin')).divide(2).rename('tmean')
    
    # 2. Calculate Saturation Vapor Pressure (es) in Pascals
    # Tetens formula: es = 610.78 * exp((17.27 * T) / (T + 237.3))
    t_exp = tmean.multiply(17.27).divide(tmean.add(237.3)).exp()
    es = t_exp.multiply(610.78).rename('es')
    
    # 3. Calculate VPD (es - ea). Daymet 'vp' is already in Pascals
    vpd = es.subtract(image.select('vp')).rename('vpd')
    
    # Combine all these new calculated bands into the image
    final_image = image.addBands([tmean, vpd])
    
    reduced = image.reduceRegions(
        collection=grid_5km,
        reducer=ee.Reducer.mean(), # Calculates the average weather inside the 5km box
        scale=1000 # Samples the image at 1km resolution to calculate the mean
    )
    return reduced.map(lambda feature: feature.set('Date', date))

# 5. The Batch Orchestration Loop
start_year = config['start_year']
end_year = config['end_year']

print(f"\nOrchestrating batch exports from {start_year} to {end_year}...")

for year in range(start_year, end_year + 1):
    
    # Define exact dates for this specific loop
    yearly_start = f"{year}-01-01"
    yearly_end = f"{year}-12-31"
    
    # Filter the main collection to just this year
    yearly_collection = main_climate_collection.filterDate(yearly_start, yearly_end)
    
    # Apply the extraction function (FIX 2)
    # This was commented out previously. This actually executes the math and 
    # creates the tabular dataset for the export task.
    tabular_data = yearly_collection.map(extract_tabular_data).flatten()
    
    # Dynamically name the file
    task_name = f"Daymet_data_{year}"
    
    # Create the export task
    export_task = ee.batch.Export.table.toDrive(
        collection=tabular_data,
        description=task_name,
        folder='Masters_Project_Data',
        fileFormat='CSV',
        selectors=['Grid_ID', 'Date', 'tmax', 'tmin', 'tmean', 'prcp', 'vpd'] # Only export the columns we need
    )
    
    # Send to Google's servers
    export_task.start()
    print(f"[{year}] Task successfully sent to GEE: {task_name}")

print("\nAll tasks submitted! Check the Earth Engine Code Editor 'Tasks' tab to monitor progress.")