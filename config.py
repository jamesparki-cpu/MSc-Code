# config.py

# File Paths
RAW_MOSQUITO_DATA = ""
OUTPUT_DATA_PATH = ""

# Timeframe
START_DATE = ''
END_DATE = ''

# Google Earth Engine Dataset IDs
GEE_DATASETS = {
    'climate_daymet': 'NASA/ORNL/DAYMET_V4',
    'landcover_nlcd': 'USGS/NLCD_RELEASES/2021_REL/NLCD',
    'vegetation_modis': 'MODIS/061/MOD13Q1'
}

# Grid / Spatial settings
SPATIAL_BUFFER_METERS = 0000  # For the spatial thinning 