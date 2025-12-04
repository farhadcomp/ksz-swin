import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch


folder_name = f"plots_Final_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
folder_path = os.path.join(os.getcwd(), folder_name)
os.makedirs(folder_path, exist_ok=True)
BEST_MODEL_PATH = os.path.join(folder_path, "best_model.pth")
LAPLACE_STATE_PATH = os.path.join(folder_path, "fitted_laplace.pth")


def save_plot(fig, filename):
    save_path = os.path.join(folder_path, filename)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
# Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    with h5py.File(file_path, "r") as h5f:
        ydata = h5f["tau_values"][:]

    # scaler = StandardScaler()
    # rescaled_tau = scaler.fit_transform(ydata.reshape(-1, 1)).flatten()
    # ydata_tensor = torch.tensor(rescaled_tau, dtype=torch.float32)
    # return ydata_tensor, scaler
    return ydata

file_path = "/home/laplante/data/ksz/all_snapshots.hdf5"
# ydata, tau_scaler = load_and_preprocess_data(file_path)
ydata = load_and_preprocess_data(file_path)

fig , ax = plt.subplots()
n, bins, patches = ax.hist(ydata, bins=15, density=True, 
                           alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(x=0.054, color='red', linestyle='--', linewidth=2, label='Planck 2020 Mean')
ax.set_xlabel(r"Optical Depth ($\tau$)")
ax.set_ylabel("Probability Density")
ax.set_title(r"Distribution of $\tau$ Values in Training Set")
ax.legend()

save_plot(fig, "tau_histogram_normalized.pdf")