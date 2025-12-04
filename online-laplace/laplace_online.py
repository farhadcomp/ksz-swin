import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import torch.optim as optim
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
from torch.optim.lr_scheduler import CosineAnnealingLR


from laplace import Laplace, marglik_training
from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.asdl import AsdlGGN

start_time = time.time()

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a unique folder for saving plots
folder_name = f"plots_Laplace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
folder_path = os.path.join(os.getcwd(), folder_name)
os.makedirs(folder_path, exist_ok=True)
BEST_MODEL_PATH = os.path.join(folder_path, "best_model.pth")


def save_plot(fig, filename):
    save_path = os.path.join(folder_path, filename)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)


# Data loading and preprocessing
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
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((224, 224), antialias=True)]
            )

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        image = self.xdata[idx]
        label = self.ydata[idx]

        image = self.transform(image)

        return image, label.unsqueeze(-1)


X_temp, X_test, y_temp, y_test = train_test_split(
    xdata, ydata, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

train_dataset = KSZDataset(X_train, y_train, is_train=True)
test_dataset = KSZDataset(X_test, y_test, is_train=False)
val_dataset = KSZDataset(X_val, y_val, is_train=False)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.swin_model = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        for param in self.swin_model.parameters():
            param.requires_grad = False
        # for name, param in self.swin_model.named_parameters():
        #     if "layers.3" in name or "norm" in name:
        #         param.requires_grad = True
        # self.head = nn.Sequential(
        #     nn.Linear(self.swin_model.num_features, 32),
        #     nn.Linear(32, 170),
        #     nn.ReLU(),
        #     nn.Dropout(0.24),
        #     nn.Linear(170, 44), nn.ReLU(),
        #     nn.Linear(44, 63), nn.ReLU(),
        #     nn.Linear(63, 1)
        # )

        self.head = nn.Sequential(
            nn.Linear(self.swin_model.num_features, 32),
            nn.Linear(32, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, 32), nn.ReLU(),
            # # # nn.Linear(128, 64), nn.ReLU(),
            # # nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        features = self.swin_model(x).last_hidden_state.mean(dim=1)
        return self.head(features)


model = CombinedModel().to(device)

n_epochs = 500

laplace_kwargs = {"subset_of_weights": "last_layer", "hessian_structure": "diag"}

# model = get_model()
# la, model, margliks, losses = marglik_training(
#     model=model,
#     train_loader=train_dataloader,
#     likelihood="regression",
#     hessian_structure="full",
#     marglik_frequency=10,
#     n_epochs_burnin=10,
#     progress_bar=True,
#     scheduler_cls=CosineAnnealingLR,
#     n_epochs=n_epochs,
#     optimizer_kwargs={"lr": 1e-5},
#     scheduler_kwargs={'T_max': n_epochs},
# )

la, model, margliks, losses = marglik_training(
    model=model,
    train_loader=train_dataloader,
    likelihood="regression",
    hessian_structure="full",
    marglik_frequency=10,
    backend=AsdlGGN,
    lr_hyp=0.01,
    n_epochs_burnin=20,
    progress_bar=True,
    # scheduler_cls=CosineAnnealingLR,
    n_epochs=n_epochs,
    optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-3},
    # prior_structure="scalar",
    # scheduler_kwargs={'T_max': n_epochs},
)

y_mean_scaled_list, y_var_scaled_list, true_values_scaled_list = [], [], []
with torch.no_grad():
    for inputs, targets in tqdm.tqdm(test_dataloader):
        inputs = inputs.to(device)
        mean, var = la(inputs)
        y_mean_scaled_list.append(mean.squeeze().cpu())
        y_var_scaled_list.append(var.squeeze().cpu())
        true_values_scaled_list.append(targets)

# f_mu, f_var = la(X_test)
# f_mu = f_mu.squeeze().detach().cpu().numpy()
# f_sigma = f_var.squeeze().sqrt().cpu().numpy()
# pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

y_mean_scaled = torch.cat(y_mean_scaled_list).numpy()
y_var_scaled = torch.cat(y_var_scaled_list).numpy()
true_values_scaled = torch.cat(true_values_scaled_list).squeeze().numpy()

noise_variance = la.sigma_noise.item() ** 2
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



print("\n--- Final Test Metrics ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"Chi-Squared Metric: {chi_squared_metric:.2f}")

# Plotting...
# Plot the training loss
# fig, ax = plt.subplots()
# ax.plot(losses, label="Training Loss")
# ax.set_title("Training Loss per Epoch")
# ax.set_xlabel("Epoch")
# ax.set_ylabel("Loss (MSE)")
# ax.set_yscale("log")
# ax.legend()
# # ax.grid(True)
# plt.show()

# save_plot(fig, "training_loss_plot.png")

# Create the plot
fig, ax1 = plt.subplots()

# Plot training loss (MSE) on the left y-axis
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss (MSE)", color="tab:blue")
ax1.plot(losses, color="tab:blue", label="Training Loss")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Plot log marginal likelihood on the right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel("Log Marginal Likelihood", color="tab:red")
ax2.plot(margliks, color="tab:red", label="Log MargLik")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.tight_layout()
plt.title("Training Loss vs. Marginal Likelihood")
# plt.show()

# Save the plot
save_plot(fig, "overfitting_check.pdf")


fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(
    true_values, y_mean, alpha=0.7, color="blue", s=10, label=f"$R^2 = {r2:.3f}$"
)
min_val, max_val = min(np.min(true_values), np.min(y_mean)), max(
    np.max(true_values), np.max(y_mean)
)
ax_scatter.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Fit")
formatter = ticker.StrMethodFormatter('{x:.3f}')
ax_scatter.yaxis.set_major_formatter(formatter)
ax_scatter.xaxis.set_major_formatter(formatter)
ax_scatter.set_title("Predicted vs. Actual Values (Laplace)")
ax_scatter.set_xlabel("True Values")
ax_scatter.set_ylabel("Predicted Values")
ax_scatter.legend()
save_plot(fig_scatter, "Scatterplot_Laplace.pdf")

fig_error, ax_error = plt.subplots()
ax_error.errorbar(true_values, y_mean, yerr=y_sigma, fmt="o", alpha=0.6, capsize=3)
ax_error.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Fit")
formatter = ticker.StrMethodFormatter('{x:.3f}')
ax_error.yaxis.set_major_formatter(formatter)
ax_error.xaxis.set_major_formatter(formatter)

# my_ticks = np.arange(min_val, max_val, 0.0025) 
# ax_error.set_xticks(my_ticks)
# ax_error.set_yticks(my_ticks)
ax_error.set_title("Prediction vs. True with Uncertainty (Laplace)")
ax_error.set_xlabel("True Values")
ax_error.set_ylabel("Predicted Values")
ax_error.legend()
save_plot(fig_error, "Prediction_Errorbar_Laplace.pdf")

end_time = time.time()
running_time = (end_time - start_time) / 60
print(f"\nTotal running time: {running_time:.4f} minutes")

n_epochs_burnin = 10
print(f"\nWith n_epochs_burnin= {n_epochs_burnin}")
