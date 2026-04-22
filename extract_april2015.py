import ee
import pandas as pd
import json

# 1. Load your configuration
print("Loading config...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# 2. Initialize using the config variable
ee.Initialize(project=config['gcp_project_id']) 
print("Connected to Earth Engine.")

# 3. Use the config to define the boundary
states = ee.FeatureCollection("TIGER/2018/States")
target_geometry = states.filter(ee.Filter.eq('NAME', config['target_state'])).first().geometry()

# 4. Use the config for your date filters
dataset = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
    .filterBounds(target_geometry) \
    .filterDate(config['start_date'], config['end_date']) \
    .select(['tmax', 'tmin'])

# 5. Calculate the daily average
def get_daily_mean(image):
    mean_temp = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=target_geometry,
        scale=config['scale_meters'], # Using config here too!
        maxPixels=1e9 
    )
    
    date = image.date().format('YYYY-MM-dd')
    
    return ee.Feature(None, {
        'date': date,
        'tmax': mean_temp.get('tmax'),
        'tmin': mean_temp.get('tmin')
    })

print(f"Calculating data for {config['target_state']} from {config['start_date']} to {config['end_date']}...")
daily_temps = dataset.map(get_daily_mean)

# 6. Download and format
print("Downloading results...")
data_list = daily_temps.reduceColumns(
    reducer=ee.Reducer.toList(3),
    selectors=['date', 'tmax', 'tmin']
).values().get(0).getInfo()

df = pd.DataFrame(data_list, columns=['Date', 'Max_Temp_C', 'Min_Temp_C'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)

print(f"\n--- {config['target_state']} Daily Temperatures ---")
print(df)

df.to_csv('florida_april_2015.csv')

print("Data successfully saved to florida_april_2015.csv!")
