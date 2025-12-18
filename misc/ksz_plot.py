import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py

# --- 1. Load a Sample kSZ Map ---
# Replace with your actual file path
file_path = "/home/laplante/data/ksz/all_snapshots.hdf5"

# If file not found, generate a dummy map for demonstration
try:
    with h5py.File(file_path, "r") as f:
        # Assuming maps are stored as images in a dataset, e.g., 'maps'
        # Load the first map
        ksZ_map = f["ksz_maps"][0] 
except:
    print("File not found or structure differs. Using dummy data.")
    ksZ_map = np.random.normal(0, 1, (224, 224)) # 224x224 typical for Swin

# --- 2. Setup Plot Parameters ---
# Map corresponds to ~20 degrees field of view (from Section 2.2) [cite: 86]
extent = [-10, 10, -10, 10] # degrees (approx, based on 20 deg width)
window_size = 56 # Example: 224 pixels / 4 windows = 56 pixels per window
shift_size = window_size // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- 3. Helper Function to Draw Grids ---
def draw_window_grid(ax, shift=False):
    # Plot the map
    im = ax.imshow(ksZ_map, extent=extent, cmap='RdBu_r', origin='lower')
    
    # Add Grid Lines (Windows)
    # Convert physical extent to array indices logic for grid placement
    # We draw lines based on the 4x4 window partition described in caption [cite: 156]
    
    # For visualization, we just draw the grid lines over the extent
    x_grid = np.linspace(extent[0], extent[1], 5) # 4 windows -> 5 edges
    y_grid = np.linspace(extent[2], extent[3], 5)
    
    if shift:
        # Shift grid by half window size
        step = (extent[1] - extent[0]) / 4
        x_grid = x_grid - (step/2)
        y_grid = y_grid - (step/2)
    
    # Draw vertical lines
    for x in x_grid:
        if extent[0] <= x <= extent[1]: # Only draw if inside plot
            ax.axvline(x, color='black', linewidth=1.5)
            
    # Draw horizontal lines
    for y in y_grid:
        if extent[2] <= y <= extent[3]:
            ax.axhline(y, color='black', linewidth=1.5)

    # --- THE FIX: ADD UNITS ---
    ax.set_xlabel(r"$\theta_x$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\theta_y$ [deg]", fontsize=12)
    return im

# --- 4. Plot Panel A (Layer l: Regular Partition) ---
draw_window_grid(axes[0], shift=False)
axes[0].set_title("Layer $l$ (Regular Partitioning)")

# --- 5. Plot Panel B (Layer l+1: Shifted Partition) ---
draw_window_grid(axes[1], shift=True)
axes[1].set_title("Layer $l+1$ (Shifted Partitioning)")

# Clean up
plt.tight_layout()
plt.savefig("Figure3_with_units.pdf", dpi=300)
plt.show()