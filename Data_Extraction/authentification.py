import ee 
import geemap
import pandas as pd

# This opens a web browser to log into your Google Account
ee.Authenticate() 

# This starts the Earth Engine session
ee.Initialize(project='msc-dis-data')