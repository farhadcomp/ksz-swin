import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
from transformers import SwinModel
import torch
from datetime import datetime
import time
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Laplace import
from laplace import Laplace

# --- Initial Setup ---
start_time = time.time()
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

folder_name = f"plots_Final_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
folder_path = os.path.join(os.getcwd(), folder_name)
os.makedirs(folder_path, exist_ok=True)
BEST_MODEL_PATH = os.path.join(folder_path, "best_model.pth")
LAPLACE_STATE_PATH = os.path.join(folder_path, "fitted_laplace.pth")


def save_plot(fig, filename):
    save_path = os.path.join(folder_path, filename)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    with h5py.File(file_path, "r") as h5f:
        xdata = h5f["ksz_maps"][:]
        ydata = h5f["tau_values"][:]
    tcmb = 2.75
    rescaled_map = xdata * tcmb * 1e6
    scaler = StandardScaler()
    rescaled_tau = scaler.fit_transform(ydata.reshape(-1, 1)).flatten()
    xdata_tensor = torch.tensor(rescaled_map, dtype=torch.float32)
    ydata_tensor = torch.tensor(rescaled_tau, dtype=torch.float32)
    xdata_tensor = xdata_tensor.unsqueeze(1)
    return xdata_tensor, ydata_tensor, scaler

file_path = "/home/laplante/data/ksz/all_snapshots.hdf5"
xdata, ydata, tau_scaler = load_and_preprocess_data(file_path)

class KSZDataset(Dataset):
    def __init__(self, xdata, ydata, is_train=False):
        self.xdata = xdata
        self.ydata = ydata
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True)
            ])
    def __len__(self):
        return len(self.xdata)
    def __getitem__(self, idx):
        image = self.xdata[idx]
        label = self.ydata[idx]
        image = self.transform(image)
        return image, label

X_temp, X_test, y_temp, y_test = train_test_split(xdata, ydata, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

train_dataset = KSZDataset(X_train, y_train, is_train=True)
val_dataset = KSZDataset(X_val, y_val, is_train=False)
test_dataset = KSZDataset(X_test, y_test, is_train=False)

# --- 2. Model Definition ---
class CombinedModel(nn.Module):
    def __init__(self, head_dims, dropout_rate, num_unfrozen_blocks=1):
        super(CombinedModel, self).__init__()
        self.swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
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
        x = x.repeat(1, 3, 1, 1)
        features = self.swin_model(x).last_hidden_state.mean(dim=1)
        return self.head(features).squeeze(-1)

# --- 3. Final Model Training ---
print("--- Starting Final Model Training with Best Hyperparameters ---")

# ✅ Define the best parameters you found
best_params = {
    'lr': 9.792998174278061e-05, 
    'head_dims': [95, 210, 202], 
    'dropout_rate': 0.11072841603148399, 
    'weight_decay': 0.006559148317250172, 
    'num_unfrozen_blocks': 2
}
print("Using parameters:", best_params)

# Create model, dataloaders, and optimizer
final_model = CombinedModel(
    head_dims=best_params["head_dims"],
    dropout_rate=best_params["dropout_rate"],
    num_unfrozen_blocks=best_params["num_unfrozen_blocks"]
).to(device)

optimizer = optim.Adam(
    final_model.parameters(), 
    lr=best_params['lr'], 
    weight_decay=best_params['weight_decay']
)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
criterion = nn.MSELoss()

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Standard Training Loop
num_epochs = 500 
patience = 30
epochs_without_improve = 0
best_val_loss = float('inf')
val_losses = []
train_losses = []

for epoch in range(num_epochs):
    final_model.train()
    running_train_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
    avg_train_loss = running_train_loss /len(train_dataloader)
    train_losses.append(avg_train_loss)
    # Validation
    final_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = final_model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Validation MSE: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(final_model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> New best model saved to {BEST_MODEL_PATH}")
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1
    
    if epochs_without_improve >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Load the best performing model before applying Laplace
final_model.load_state_dict(torch.load(BEST_MODEL_PATH))
print("Best model weights loaded for Laplace.")

# Applying Laplace Approximation
print("\n Applying Laplace Approximation")
la = Laplace(final_model, "regression", subset_of_weights="last_layer", hessian_structure="full")
la.fit(train_dataloader)

# You can optionally optimize Laplace hyperparameters here if needed
print(" Optimizing Laplace Hyperparameters")
log_prior, log_sigma = (
    torch.ones(1, requires_grad=True, device=device),
    torch.ones(1, requires_grad=True, device=device),
)
hyper_optimizer = optim.Adam([log_prior, log_sigma], lr=1e-2)

for i in range(100):
    hyper_optimizer.zero_grad()
    # The marglik_training function minimizes this, here we do it manually
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

print(f"Optimized sigma={la.sigma_noise.item():.2f}", f"prior precision={la.prior_precision.item():.2f}")


# Save the fitted Laplace Object
torch.save(la.state_dict(), LAPLACE_STATE_PATH)
print(f"Fitted Laplace model saved to {LAPLACE_STATE_PATH}")

# Load the saved laplace object
model_for_loading = CombinedModel(
    head_dims=best_params["head_dims"],
    dropout_rate=best_params["dropout_rate"],
    num_unfrozen_blocks=best_params["num_unfrozen_blocks"]
).to(device)
model_for_loading.load_state_dict(torch.load(BEST_MODEL_PATH))
model_for_loading.eval()

la_loaded = Laplace(model_for_loading, "regression", subset_of_weights="last_layer", hessian_structure="full")

la_loaded.load_state_dict(torch.load(LAPLACE_STATE_PATH))

print(f"Successfully loaded fitted Laplace model from {LAPLACE_STATE_PATH}")
# Evaluation and Plotting
print("\n Evaluating Final Model on Test Set")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
y_mean_scaled_list, y_var_scaled_list, true_values_scaled_list = [], [], []
with torch.no_grad():
    for inputs, targets in tqdm.tqdm(test_dataloader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)
        mean, var = la_loaded(inputs)
        y_mean_scaled_list.append(mean.squeeze().cpu())
        y_var_scaled_list.append(var.squeeze().cpu())
        true_values_scaled_list.append(targets.cpu())

y_mean_scaled = torch.cat(y_mean_scaled_list).numpy()
y_var_scaled = torch.cat(y_var_scaled_list).numpy()
true_values_scaled = torch.cat(true_values_scaled_list).squeeze().numpy()

noise_variance = la_loaded.sigma_noise.item() ** 2
y_total_var_scaled = y_var_scaled + noise_variance 
y_sigma_scaled = np.sqrt(y_total_var_scaled)

scaled_mean_2d = y_mean_scaled.reshape(-1, 1)
scaled_true_2d = true_values_scaled.reshape(-1, 1)

y_mean = tau_scaler.inverse_transform(scaled_mean_2d).flatten()
true_values = tau_scaler.inverse_transform(scaled_true_2d).flatten()

y_sigma = y_sigma_scaled * tau_scaler.scale_
# Final metrics and plots
mae = mean_absolute_error(true_values, y_mean)
rmse = np.sqrt(mean_squared_error(true_values, y_mean))
r2 = r2_score(true_values, y_mean)
pearson, _ = pearsonr(true_values, y_mean)

# 
chi_squared_metric = np.sum(((true_values - y_mean) / y_sigma) ** 2)


print("\n Final Test Metrics")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"Chi-Squared Metric: {chi_squared_metric:.2f}")

fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(true_values, y_mean, alpha=0.7, color='blue', s=10, label=f'$R^2 = {r2:.3f}$')
min_val, max_val = min(np.min(true_values), np.min(y_mean)), max(np.max(true_values), np.max(y_mean))
ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Fit")
formatter = ticker.StrMethodFormatter('{x:.3f}')
ax_scatter.yaxis.set_major_formatter(formatter)
ax_scatter.xaxis.set_major_formatter(formatter)
ax_scatter.set_title("Predicted vs. Actual Values (Laplace)")
ax_scatter.set_xlabel("True Values")
ax_scatter.set_ylabel("Predicted Values")
ax_scatter.legend()
save_plot(fig_scatter, "Scatterplot_Laplace.pdf")

fig_error, ax_error = plt.subplots()
ax_error.errorbar(true_values, y_mean, yerr=y_sigma, fmt='o', alpha=0.6, capsize=3)
ax_error.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Fit")
formatter = ticker.StrMethodFormatter('{x:.3f}')
ax_error.yaxis.set_major_formatter(formatter)
ax_error.xaxis.set_major_formatter(formatter)
ax_error.set_title("Prediction vs. True with Uncertainty (Laplace)")
ax_error.set_xlabel("True Values")
ax_error.set_ylabel("Predicted Values")
ax_error.legend()
save_plot(fig_error, "Prediction_Errorbar_Laplace.pdf")

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(train_losses, label="Training Loss")
ax_loss.plot(val_losses, linestyle='--', label="Validation Loss")
ax_loss.set_yscale("log")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("MSE Loss")
ax_loss.set_title("Training and Validation Loss")
ax_loss.legend()
# ax_loss.grid(True)
save_plot(fig_loss, "Train_Val_Loss_Laplace.pdf")

end_time = time.time()
running_time = (end_time - start_time) / 60
print(f"\nTotal running time: {running_time:.2f} minutes")