import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "/home/laplante/data/ksz/all_snapshots.hdf5"

print("Loading all tau values...")
with h5py.File(DATA_PATH, "r") as hf:
    # Load ALL tau values from the dataset
    all_taus = hf["tau_values"][:]

print(f"Fitting scaler on {len(all_taus)} samples...")
scaler = StandardScaler()
scaler.fit(all_taus.reshape(-1, 1))

# Verify it looks correct (mean should be ~0.054)
print(f"Scaler Mean: {scaler.mean_[0]:.4f}")
print(f"Scaler Scale (Std): {scaler.scale_[0]:.4f}")

# Save to disk
joblib.dump(scaler, 'tau_scaler.pkl')
print("Scaler saved to 'tau_scaler.pkl'")