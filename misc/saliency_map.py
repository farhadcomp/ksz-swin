import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from astropy.cosmology import Planck18
from astropy import units
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import SwinModel
import joblib


MODEL_PATH = "/home/farhadik/ksz-swin/offline-laplace/plots_Final_Run_20251204_120146/best_model.pth" 
DATA_PATH = "/home/laplante/data/ksz/all_snapshots.hdf5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedModel(nn.Module):
    def __init__(self, head_dims, dropout_rate, num_unfrozen_blocks=1):
        super(CombinedModel, self).__init__()
        # Load the base Swin Transformer
        self.swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Freeze all parameters initially
        for param in self.swin_model.parameters():
            param.requires_grad = False
            
        # Unfreeze specific blocks
        if num_unfrozen_blocks > 0:
            for i in range(num_unfrozen_blocks):
                layer_to_unfreeze = self.swin_model.num_layers - 1 - i
                for name, param in self.swin_model.named_parameters():
                    if f"layers.{layer_to_unfreeze}" in name or "norm" in name:
                          param.requires_grad = True
        
        # Define the Regression Head
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
        # IMPORTANT: Input x is expected to be (Batch, 1, H, W)
        # The model converts it to 3 channels here:
        x = x.repeat(1, 3, 1, 1) 
        features = self.swin_model(x).last_hidden_state.mean(dim=1)
        return self.head(features).squeeze(-1)

# --- 2. Setup Device and Parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your Best Parameters (from your snippet)
best_params = {
    'head_dims': [95, 210, 202], 
    'dropout_rate': 0.11072841603148399, 
    'num_unfrozen_blocks': 2
}

# --- 3. Load the Model ---
# Initialize the model structure
model = CombinedModel(
    head_dims=best_params["head_dims"],
    dropout_rate=best_params["dropout_rate"],
    num_unfrozen_blocks=best_params["num_unfrozen_blocks"]
).to(device)

np.random.seed(42)
torch.manual_seed(42)

num_samples = 1000
indices = np.arange(num_samples)
temp_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42)

index_to_test = test_indices[1]

try:
    print("Loading saved scaler...")
    scaler = joblib.load('tau_scaler.pkl')
except FileNotFoundError:
    print("Error: 'tau_scaler.pkl' not found. Please run the scaler setup script first.")
    sys.exit(1)
    
# We load a single map to test
try:
    with h5py.File(DATA_PATH, "r") as hf:
        # Load one map (e.g., the first one in the test set)
        raw_map = hf["ksz_maps"][index_to_test:index_to_test+1]
        tcmb = 2.75
        rescaled_map = raw_map * tcmb * 1e6
        # scaler = StandardScaler()
        true_tau = hf["tau_values"][index_to_test]
        print("true_tau: ", true_tau)
        # rescaled_tau = scaler.fit_transform(true_tau.reshape(-1, 1)).flatten()
        xdata_tensor = torch.tensor(rescaled_map, dtype=torch.float32)
        # print(xdata_tensor.shape)
        # ydata_tensor = torch.tensor(rescaled_tau, dtype=torch.float32)
        # xdata_tensor = xdata_tensor.unsqueeze(0)
        # print(xdata_tensor.shape)
except Exception as e:
    print(f"Error loading data: {e}")
    # Dummy data for testing the script
    raw_map = np.random.normal(0, 1, (224, 224))
    true_tau = 0.054
    
eval_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True)
])

# Apply the resizing transform
input_tensor_resized = eval_transform(xdata_tensor)

# print(input_tensor_resized.shape)

# Add Batch Dimension -> (1, 1, 224, 224)
input_batch = input_tensor_resized.unsqueeze(0).to(device)

# print(input_batch.shape)

# if input_batch.shape[1] == 1:
#      input_batch = input_batch.repeat(1, 3, 1, 1)
        
# IMPORTANT: We need gradients for the input image to compute saliency
input_batch.requires_grad_()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH} not found. Using random weights for demonstration.")
    
model.eval()

# sys.exit(0)


# Forward pass
output = model(input_batch)

print("output: ", output)
# print("ydata: ", ydata_tensor)

predicted_tau = output.item()

print("predicted_tau: ", predicted_tau)

# sys.exit(0)
# Backward pass
model.zero_grad()
output.backward()

# Get gradients relative to input
gradients = input_batch.grad.data.cpu().numpy()[0]

# print(gradients.shape)

# sys.exit(0)

# Take the maximum magnitude across channels (if 3 channels) to get a 2D map
if gradients.shape[0] == 3:
    saliency = np.max(np.abs(gradients), axis=0)
else:
    saliency = np.abs(gradients[0])

# Normalize saliency for visualization (0 to 1)
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

input_image_np = input_tensor_resized.squeeze().cpu().numpy()

print("predicted_tau: ", predicted_tau)
predicted_tau = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 1)).flatten()[0]
print("predicted_tau: ", predicted_tau)
# Physics scaling for axes (reusing your previous code logic)
lbox = 2 * units.Gpc / Planck18.h
z = 15.0
a = 1 / (1 + z)
r_phys = a * lbox
theta = float(r_phys / Planck18.angular_diameter_distance(z))
theta_deg = theta * 180 / np.pi
extent_args = [-theta_deg / 2, theta_deg / 2, -theta_deg / 2, theta_deg / 2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot A: Original kSZ Map
tmax = np.max(np.abs(input_image_np))
im1 = ax[0].imshow(input_image_np, cmap="RdBu", vmin=-tmax, vmax=tmax, 
                   extent=extent_args, origin='lower')
ax[0].set_title(f"Original kSZ Map (True $\\tau$ = {true_tau:.4f})")
ax[0].set_xlabel(r"$\theta_x$ [deg]")
ax[0].set_ylabel(r"$\theta_y$ [deg]")
plt.colorbar(im1, ax=ax[0], label=r"$\mu$K", fraction=0.046, pad=0.04)

# Plot B: Saliency Map
# We use 'hot' or 'inferno' to show intensity
im2 = ax[1].imshow(saliency, cmap="hot", extent=extent_args, origin='lower')
ax[1].set_title(f"Saliency Map(Predicted $\\tau$ = {predicted_tau:.4f})")
ax[1].set_xlabel(r"$\theta_x$ [deg]")
ax[1].set_ylabel(r"$\theta_y$ [deg]")
cbar = plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
cbar.set_label("Relative Importance", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("Figure_Saliency_Map.pdf", dpi=300)
plt.show()