import rasterio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

file_path = "FL_Temp_2015-04-01_to_2015-05-01_5000m.tif"
print(f"Loading {file_path} into memory...")

with rasterio.open(file_path) as src:
    # 1. Read the ENTIRE file into a 3D NumPy array
    # Shape will be: (60 layers, height, width)
    data = src.read()
    
# 2. Separate the Max and Min temperatures
# Earth Engine exported them in order: TMAX, TMIN, TMAX, TMIN...
# We use Python array slicing [start:stop:step] to grab every other layer
tmax_data = data[0::2].astype(float)  # Starts at index 0, skips by 2 (All TMAX layers)
tmin_data = data[1::2].astype(float)  # Starts at index 1, skips by 2 (All TMIN layers)
tmax_data[tmax_data > 100] = np.nan
tmin_data[tmin_data > 100] = np.nan

num_days = tmax_data.shape[0]

# 3. Set up the canvas and make room at the bottom for the slider
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2) 

# 4. Draw the initial map (Day 1 Max Temp)
# We use 'imshow' here instead of rasterio.show for faster dynamic updating
img = ax.imshow(tmax_data[0], cmap='inferno', vmin=10, vmax=35)

# Add a color scale bar to the side
fig.colorbar(img, ax=ax, label='Max Temp (C)')
ax.set_title("Florida 5km Grid: Max Temp - Day 1")
ax.axis('off') # Hides the axis numbers for a cleaner map look

# 5. Build the Slider UI
# [left, bottom, width, height] positions the slider on the window
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(
    ax=ax_slider, 
    label='Day of April', 
    valmin=1, 
    valmax=num_days, 
    valinit=1, 
    valstep=1
)

# 6. Create the update function
# This runs every single time you drag the slider
def update(val):
    day = int(slider.val)
    # Update the map pixels to the new day (Subtract 1 because Python is 0-indexed)
    img.set_data(tmax_data[day - 1])
    # Update the title
    ax.set_title(f"Florida 5km Grid: Max Temp - Day {day}")
    # Force the canvas to redraw
    fig.canvas.draw_idle()

# Tell the slider to trigger the update function when moved
slider.on_changed(update)

print("Opening interactive viewer...")
plt.show()
