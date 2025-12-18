import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py
import joblib
from transformers import SwinModel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For progress bar
import torchvision.transforms as transforms


# --- 1. CONFIGURATION ---
MODEL_PATH = "plots_Final_Run_YOUR_TIMESTAMP/best_model.pth"
DATA_PATH = "/home/laplante/data/ksz/all_snapshots.hdf5"
SCALER_PATH = "tau_scaler.pkl" 
INDEX_TO_TEST = 521             
N_MASKS = 2000                  # Higher = smoother map (1000-5000 is good)
MASK_PROB = 0.5                 # Probability of a pixel being visible
GRID_SIZE = (7, 7)              # Low-res grid size for masks (7x7 or 14x14 works well for Swin)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION (Must match your training) ---
class CombinedModel(nn.Module):
    def __init__(self, head_dims, dropout_rate, num_unfrozen_blocks=1):
        super(CombinedModel, self).__init__()
        self.swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Freezing/Unfreezing logic (simplified for inference loading)
        for param in self.swin_model.parameters():
            param.requires_grad = False
            
        layers = []
        in_dim = self.swin_model.num_features
        for i, out_dim in enumerate(head_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # Handle single channel input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.swin_model(x).last_hidden_state.mean(dim=1)
        return self.head(features).squeeze(-1)

# --- 3. RISE UTILITY FUNCTION ---
def generate_rise_map(model, input_tensor, n_masks=1000, p=0.5, grid_size=(7, 7)):
    """
    Generates a saliency map using Randomized Input Sampling (RISE).
    """
    # Get image size
    _, _, H, W = input_tensor.shape
    
    # 1. Generate random binary masks (low res)
    # Shape: (N, 1, h, w)
    low_res_masks = torch.rand(n_masks, 1, *grid_size, device=device)
    low_res_masks = (low_res_masks < p).float()
    
    # 2. Upsample masks to image size (H, W) smoothly
    # Shape: (N, 1, H, W)
    masks = torch.nn.functional.interpolate(low_res_masks, size=(H, W), mode='bilinear', align_corners=False)
    
    # 3. Accumulate weighted maps
    saliency = torch.zeros(1, 1, H, W, device=device)
    
    # Run in batches to save GPU memory
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, n_masks, batch_size), desc="Running RISE"):
            # Batch of masks
            m_batch = masks[i : min(i+batch_size, n_masks)]
            actual_batch_size = m_batch.shape[0]
            
            # Mask the input: Input * Mask
            # We expand input to match batch size
            masked_input = input_tensor.expand(actual_batch_size, -1, -1, -1) * m_batch
            
            # Run Model
            outputs = model(masked_input) # outputs are (Batch_Size,)
            
            # Weight masks by the predicted score
            # Reshape score to (Batch, 1, 1, 1) to multiply mask
            scores = outputs.view(actual_batch_size, 1, 1, 1)
            
            # Add to total saliency: sum(Mask_i * Score_i)
            saliency += (scores * m_batch).sum(dim=0, keepdim=True)
            
    # Normalize
    saliency = saliency / n_masks
    return saliency.squeeze().cpu().numpy()

# --- 4. MAIN EXECUTION ---
# Load Scaler
try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    print("Please run the scaler fitting script first!")
    import sys; sys.exit(1)

# Load Data
with h5py.File(DATA_PATH, "r") as hf:
    raw_map = hf["ksz_maps"][INDEX_TO_TEST]
    true_tau = hf["tau_values"][INDEX_TO_TEST]

# Preprocess
tcmb = 2.7255
rescaled_map = raw_map * tcmb * 1e6
input_tensor = torch.tensor(rescaled_map, dtype=torch.float32).unsqueeze(0) # (1, H, W)

# Resize to 224x224
resize_transform = torch.nn.Sequential(
    transforms.Resize((224, 224), antialias=True)
)
input_resized = resize_transform(input_tensor).unsqueeze(0).to(device) # (1, 1, 224, 224)

# Load Model
best_params = {'head_dims': [95, 210, 202], 'dropout_rate': 0.11, 'num_unfrozen_blocks': 2}
model = CombinedModel(**best_params).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print("Model file not found, using initialized weights.")
model.eval()

# --- RUN RISE ---
print(f"Generating RISE map for True Tau: {true_tau:.4f}...")
saliency_map = generate_rise_map(model, input_resized, n_masks=N_MASKS, p=MASK_PROB, grid_size=GRID_SIZE)

# Normalize map for plotting [0, 1]
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

# --- PLOTTING ---
from astropy.cosmology import Planck18
from astropy import units

# 1. Physics Extent Calculation (Corrected)
# We calculate the comoving box size
lbox = 2 * units.Gpc / Planck18.h 

# Convert to physical size at redshift z
z = 15.0
r_phys = (1 / (1 + z)) * lbox

# CRITICAL FIX: Convert r_phys to Mpc explicitly so units match DA(z)
r_phys_mpc = r_phys.to(units.Mpc)
da_mpc = Planck18.angular_diameter_distance(z).to(units.Mpc)

# Calculate angle in radians, then degrees
theta_rad = (r_phys_mpc / da_mpc).value
theta_deg = theta_rad * (180 / np.pi)

print(f"Calculated Field of View: {theta_deg:.2f} degrees") # Should be ~20.0 deg

extent = [-theta_deg/2, theta_deg/2, -theta_deg/2, theta_deg/2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Input
input_np = input_resized.squeeze().cpu().numpy()
tmax = np.max(np.abs(input_np))
im1 = ax[0].imshow(input_np, cmap="RdBu", vmin=-tmax, vmax=tmax, extent=extent, origin='lower')
ax[0].set_title(f"Model Input\nTrue $\\tau$={true_tau:.4f}")
ax[0].set_xlabel(r"$\theta_x$ [deg]")
ax[0].set_ylabel(r"$\theta_y$ [deg]")
plt.colorbar(im1, ax=ax[0], label=r"$\mu$K", fraction=0.046, pad=0.04)

# Saliency
im2 = ax[1].imshow(saliency_map, cmap="jet", extent=extent, origin='lower')
ax[1].set_title("RISE Importance Map\n(Perturbation Analysis)")
ax[1].set_xlabel(r"$\theta_x$ [deg]")
ax[1].set_ylabel(r"$\theta_y$ [deg]")
plt.colorbar(im2, ax=ax[1], label="Feature Importance", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("Figure_RISE_Map_Corrected.pdf", dpi=300)
plt.show()