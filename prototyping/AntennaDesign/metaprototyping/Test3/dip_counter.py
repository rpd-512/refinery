import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

CSV_PATH = "antenna_long_format_fixed.csv"

# ======================
# Tuned Parameters
# ======================
SMOOTH_WINDOW = 4
POLY_ORDER = 2
MIN_DIP_DEPTH_DB = 1.0
MIN_FREQ_SPACING_GHZ = 0.05

df = pd.read_csv(CSV_PATH)
antenna_ids = df["antenna_id"].unique()

dip_counts = {}
dip_locations = {}

# ======================
# Dip Detector
# ======================
from scipy.signal import savgol_filter
import numpy as np

def find_dips(freqs, s11):
    smooth_s11 = savgol_filter(s11, SMOOTH_WINDOW, POLY_ORDER)

    # First derivative (slope)
    d1 = np.gradient(smooth_s11)

    # Second derivative (curvature)
    d2 = np.gradient(d1)

    dips = []
    N = len(s11)

    i = 2
    while i < N - 2:
        # Look for basin start: curvature becomes strongly negative
        if d2[i] < -0.05:
            start = i
            while i < N - 2 and d2[i] < 0:
                i += 1
            end = i

            if end - start > 4:
                # Find true minimum inside basin
                basin_idx = np.argmin(s11[start:end]) + start
                depth = np.max(smooth_s11[start:end]) - s11[basin_idx]

                if depth >= MIN_DIP_DEPTH_DB:
                    dips.append((freqs[basin_idx], s11[basin_idx]))

        i += 1

    # Remove dips too close in frequency
    filtered = []
    last_f = None
    for f, v in sorted(dips):
        ghz = f / 1e9
        if last_f is None or abs(ghz - last_f) > MIN_FREQ_SPACING_GHZ:
            filtered.append((f, v))
            last_f = ghz

    return filtered, smooth_s11

# ======================
# Scan All Antennas
# ======================
for aid in antenna_ids:
    subset = df[df["antenna_id"] == aid].sort_values("frequency")

    freqs = subset["frequency"].values
    s11 = subset["s11"].values

    dips, _ = find_dips(freqs, s11)

    dip_counts[aid] = len(dips)
    dip_locations[aid] = dips

# ======================
# Find Antenna With Most Dips
# ======================
best_antenna = max(dip_counts, key=dip_counts.get)
max_dips = dip_counts[best_antenna]

print(f"\nBest antenna = {best_antenna}")
print(f"Dips found = {max_dips}")

# ======================
# Plot Result
# ======================
subset = df[df["antenna_id"] == best_antenna].sort_values("frequency")

freqs = subset["frequency"].values
s11 = subset["s11"].values

dips, smooth_s11 = find_dips(freqs, s11)

plt.figure(figsize=(10,5))

plt.plot(freqs / 1e9, s11, alpha=0.35, label="Raw S11")
plt.plot(freqs / 1e9, smooth_s11, linewidth=2, label="Smoothed")

if dips:
    dip_freqs = np.array([d[0] for d in dips]) / 1e9
    dip_vals = np.array([d[1] for d in dips])
    plt.scatter(dip_freqs, dip_vals, color="red", label="Dips")

plt.title(f"Antenna {best_antenna} — {max_dips} dips")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.legend()
plt.grid(True)
plt.show()


# ======================
# Dataset-Level Statistics
# ======================

dip_count_values = np.array(list(dip_counts.values()))

avg_dips = np.mean(dip_count_values)
std_dips = np.std(dip_count_values)

global_min_s11 = df["s11"].min()
global_min_row = df.loc[df["s11"].idxmin()]
min_freq_ghz = global_min_row["frequency"] / 1e9
min_antenna = global_min_row["antenna_id"]

print("\n===== DATASET STATS =====")
print(f"Average dips per antenna = {avg_dips:.2f}")
print(f"Std deviation of dips    = {std_dips:.2f}")
print(f"Lowest S11 ever touched  = {global_min_s11:.2f} dB")
print(f"Occurred at              = {min_freq_ghz:.3f} GHz")
print(f"Antenna ID               = {min_antenna}")


plt.figure(figsize=(9, 5))

bins = np.arange(dip_count_values.min(), dip_count_values.max() + 2)
plt.hist(dip_count_values, bins=bins, edgecolor="black", alpha=0.8)

plt.axvline(avg_dips, linestyle="--", label=f"Mean = {avg_dips:.2f}")

plt.title("Dip Count Distribution Across Antennas")
plt.xlabel("Number of Dips")
plt.ylabel("Antenna Count")
plt.legend()
plt.grid(True)

plt.show()
