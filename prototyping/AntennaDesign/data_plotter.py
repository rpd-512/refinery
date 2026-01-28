import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ======================
# CLI Arguments
# ======================
if len(sys.argv) < 3:
    print("Usage: python3 plot_antenna.py <dataset.csv> <antenna_id>")
    sys.exit(1)

CSV_PATH = sys.argv[1]
TARGET_ID = sys.argv[2]

# ======================
# Helper: Parse Array Strings
# ======================
def parse_array(text):
    text = text.strip().strip("[]")
    values = re.split(r"[\s,]+", text)
    return np.array([float(v) for v in values if v])

# ======================
# Load Dataset
# ======================
df = pd.read_csv(CSV_PATH)

# Match ID type automatically
row = df[df["id"].astype(str) == TARGET_ID]

if row.empty:
    print(f"❌ No antenna found with id = {TARGET_ID}")
    sys.exit(1)

row = row.iloc[0]

# ======================
# Parse Frequency & S11
# ======================
freqs = parse_array(row["frequencies"])
s11 = parse_array(row["s11"])

# Sort if needed
order = np.argsort(freqs)
freqs = freqs[order]
s11 = s11[order]

# ======================
# Plot
# ======================
plt.figure(figsize=(10,5))
plt.plot(freqs / 1e9, s11, linewidth=2)

plt.title(f"Antenna ID {TARGET_ID}")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.grid(True)

plt.show()
