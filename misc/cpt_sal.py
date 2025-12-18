import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py
import joblib
from transformers import SwinModel
from captum.attr import IntegratedGradients, NoiseTunnel
from astropy.cosmology import Planck18
from astropy import units
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable


MODEL_PATH = "/home/farhadik/ksz-swin/offline-laplace/plots_Final_Run_20251204_120146/best_model.pth"
DATA_PATH = "/home/laplante/data/ksz/all_snapshots.hdf5"
INDEX_TO_TEST = 947

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION ---
class CombinedModel(nn.Module):
    def __init__(self, head_dims, dropout_rate, num_unfrozen_blocks=1):
        super(CombinedModel, self).__init__()
        self.swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Freezing/Unfreezing logic matches your training
        for param in self.swin_model.parameters():
            param.requires_grad = False
            
        if num_unfrozen_blocks > 0:
            for i in range(num_unfrozen_blocks):
                layer_to_unfreeze = self.swin_model.num_layers - 1 - i
                for name, param in self.swin_model.named_parameters():
                    if f"layers.{layer_to_unfreeze}" in name or "norm" in name:
                          param.requires_grad = True
        
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
        # Input x is (Batch, 1, H, W) -> Convert to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.swin_model(x).last_hidden_state.mean(dim=1)
        return self.head(features).squeeze(-1)

# --- 3. DATA LOADING & SCALER FITTING ---
print("Initializing Data and Scaler...")
scaler = StandardScaler()

try:
    with h5py.File(DATA_PATH, "r") as hf:
        # A. Fit Scaler on EVERYTHING (Critical Step)
        print("Loading all tau values to fit global scaler...")
        all_taus = hf["tau_values"][:]
        scaler.fit(all_taus.reshape(-1, 1))
        print(f"Scaler Fitted! Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")

        # B. Load Specific Test Map
        print(f"Loading map at index {INDEX_TO_TEST}...")
        raw_map = hf["ksz_maps"][INDEX_TO_TEST] 
        true_tau = hf["tau_values"][INDEX_TO_TEST]

except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# --- 4. PREPROCESSING ---
# Convert physical units (K -> microK)
tcmb = 2.7255
rescaled_map = raw_map * tcmb * 1e6

# Convert to Tensor (1, H, W)
input_tensor = torch.tensor(rescaled_map, dtype=torch.float32).unsqueeze(0)

# Resize to 224x224 (Model Input Size)
resize_transform = transforms.Resize((224, 224), antialias=True)
input_resized = resize_transform(input_tensor).unsqueeze(0).to(device) # Add batch dim: (1, 1, 224, 224)

# Enable gradients for Captum
input_resized.requires_grad_()

# --- 5. MODEL LOADING ---
best_params = {
    'head_dims': [95, 210, 202], 
    'dropout_rate': 0.11072841603148399, 
    'num_unfrozen_blocks': 2
}

model = CombinedModel(**best_params).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH} not found. Using initialized weights.")

model.eval()

# --- 6. COMPUTE SALIENCY (INTEGRATED GRADIENTS) ---
print("Computing Saliency with Integrated Gradients + Noise Tunnel...")
ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)

# Increase nt_samples for smoother maps (e.g., 50)
attr_ig, delta = nt.attribute(input_resized, 
                              baselines=input_resized * 0, 
                              nt_type='smoothgrad_sq',
                              nt_samples=50,       # Number of noisy samples
                              stdevs=0.1,          # Noise level
                              n_steps=50,          # Integration steps
                              internal_batch_size=1,
                              return_convergence_delta=True)

# Process Attribution Map
saliency_map = attr_ig.squeeze().cpu().detach().numpy() # (224, 224)
saliency_map = np.abs(saliency_map) # Take magnitude

# Robust Normalization (Clip top 1% outliers to fix contrast)
v_max = np.percentile(saliency_map, 99)
saliency_map_clipped = np.clip(saliency_map, 0, v_max)
saliency_map_norm = (saliency_map_clipped - saliency_map_clipped.min()) / (saliency_map_clipped.max() - saliency_map_clipped.min() + 1e-8)

# Get Model Prediction
with torch.no_grad():
    pred_val = model(input_resized)
    # Unscale prediction
    pred_tau = scaler.inverse_transform(pred_val.detach().cpu().numpy().reshape(-1, 1)).flatten()[0]

print(f"True Tau: {true_tau:.5f}")
print(f"Pred Tau: {pred_tau:.5f}")

# --- 7. PLOTTING ---
# Physics Extent Calculation
lbox = 2 * units.Gpc / Planck18.h
z = 15.0
r_phys = (1 / (1 + z)) * lbox
r_phys_mpc = r_phys.to(units.Mpc)
da_mpc = Planck18.angular_diameter_distance(z).to(units.Mpc)
theta_rad = (r_phys_mpc / da_mpc).value
theta_deg = theta_rad * (180 / np.pi)
extent = [-theta_deg/2, theta_deg/2, -theta_deg/2, theta_deg/2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Panel A: Input Map
input_np = input_resized.squeeze().detach().cpu().numpy()
# tmax = np.max(np.abs(input_np))
tmax = np.amax(input_np)
tmin = np.amin(input_np)
textreme = max(tmax, np.abs(tmin))
im1 = ax[0].imshow(input_np, cmap="RdBu", vmin=-textreme, vmax=textreme, extent=extent)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
# cb = plt.colorbar(im1, cax=cax, label=r"$\mu$K")
cbl = plt.colorbar(im1, cax=cax)
cbar_label = cbl.set_label(r"$\mu$K", labelpad=-10)
my_ticks = np.arange(-7.5, 8.5, 2.5)
ax[0].set_xticks(my_ticks)
ax[0].set_yticks(my_ticks)
# ax[0].set_title(f"Model Input (Resized)\nTrue $\\tau$={true_tau:.4f}")
# ax[0].set_title(f"Model Input")
ax[0].set_xlabel(r"$\theta_x$ [deg]")
ax[0].set_ylabel(r"$\theta_y$ [deg]", labelpad=-5)
# plt.colorbar(im1, ax=ax[0], label=r"$\mu$K", fraction=0.046, pad=0.04)


# Panel B: Saliency Map
# 'inferno' provides great contrast for saliency
im2 = ax[1].imshow(saliency_map_norm, cmap="inferno", extent=extent, origin='lower')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
# cb = plt.colorbar(im2, cax=cax, label="Feature Importance")
cb = plt.colorbar(im2, cax=cax)
cbar_label = cb.set_label('Feature Importance', labelpad=1)
# cbar_label.set_position((1.05, 0.5))
ax[1].set_xticks(my_ticks)
ax[1].set_yticks(my_ticks)
# ax[1].set_title(f"Saliency Map (Integrated Gradients)\nPredicted $\\tau$={pred_tau:.4f}")
# ax[1].set_title(f"Saliency Map")
ax[1].set_xlabel(r"$\theta_x$ [deg]")
ax[1].set_ylabel(r"$\theta_y$ [deg]", labelpad=-5)
# plt.colorbar(im2, ax=ax[1], label="Feature Importance", fraction=0.046, pad=0.04)


plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
save_path = f"Figure_Saliency_Final_{INDEX_TO_TEST}.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Plot saved to {save_path}")
plt.show()