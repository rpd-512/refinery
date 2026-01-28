import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

CSV_PATH = "antenna_dataset_short.csv"
TARGET_ID = 137

# ======================
# Utilities
# ======================

def parse_array(text):
    text = text.strip().strip("[]")
    values = re.split(r"[\s,]+", text)
    return np.array([float(v) for v in values if v])

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: y_true and y_pred must match")

    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalized_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    var = np.var(y_true)
    return mse / var if var > 0 else np.nan

# ======================
# Lorentzian Model
# ======================

def lorentzian_sum(x, amp, frq, gma, n):
    amp = np.asarray(amp)
    frq = np.asarray(frq)
    gma = np.asarray(gma)

    y = np.zeros_like(x, dtype=float)

    for a, f, g in zip(amp, frq, gma):
        y += -a * g**2 / (g**2 + (x - f)**2)

    return y + n

# ======================
# Load Dataset
# ======================

df = pd.read_csv(CSV_PATH)

row = df[df["id"].astype(str) == str(TARGET_ID)]

if row.empty:
    print(f"❌ No antenna found with id = {TARGET_ID}")
    sys.exit(1)

row = row.iloc[0]

freqs = parse_array(row["frequencies"])
s11 = parse_array(row["s11"])

# Sort by frequency
order = np.argsort(freqs)
freqs = freqs[order]
s11 = s11[order]

# ======================
# Example Lorentzian Params (Synthetic Guess)
# ======================

# Example resonances — replace later with fitted params
amp = np.array([12, 18, 9])              # dip depths (dB)
frq = np.array([
    freqs.min() + 0.3*(freqs.max()-freqs.min()),
    freqs.min() + 0.6*(freqs.max()-freqs.min()),
    freqs.min() + 0.85*(freqs.max()-freqs.min())
])
gma = np.array([6e7, 1.2e8, 9e7])       # linewidths
n = -1.5                                 # baseline offset

s11_pred = lorentzian_sum(freqs, amp, frq, gma, n)

# ======================
# Compute Metrics
# ======================

mse = mean_squared_error(s11, s11_pred)
rmse = root_mean_squared_error(s11, s11_pred)
nmse = normalized_mse(s11, s11_pred)

print("\n===== FIT ERROR METRICS =====")
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"NMSE = {nmse:.6f}")

# ======================
# Plot Real vs Predicted
# ======================

plt.figure(figsize=(10, 5))

plt.plot(freqs / 1e9, s11, label="Real S11", linewidth=2)
plt.plot(freqs / 1e9, s11_pred, label="Lorentzian Model", linewidth=2, linestyle="--")

plt.title(f"Antenna ID {TARGET_ID} — Real vs Lorentzian Fit")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.legend()
plt.grid(True)
plt.show()
