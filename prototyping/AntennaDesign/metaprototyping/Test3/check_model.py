import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "antenna_long_format_fixed.csv"
MODEL_PATH = "s11_model_3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Load Dataset
# ======================
df = pd.read_csv(CSV_PATH)

# Pick random antenna_id
antenna_id = df["antenna_id"].sample(1).iloc[0]
subset = df[df["antenna_id"] == antenna_id].sort_values("frequency")

print(f"\nSelected antenna_id = {antenna_id}")

length = subset["length"].iloc[0]
width = subset["width"].iloc[0]
feed_y = subset["feed_y"].iloc[0]

freqs = subset["frequency"].values
true_s11 = subset["s11"].values

# Normalize frequency
freq_norm = freqs / 1e9

# ======================
# Model Definition
# ======================
class S11Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        return self.net(x)

model = S11Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# Build Input Batch
# ======================
X = np.stack([
    np.full_like(freq_norm, length),
    np.full_like(freq_norm, width),
    np.full_like(freq_norm, feed_y),
    freq_norm
], axis=1)

X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# ======================
# Predict S11
# ======================
with torch.no_grad():
    pred_s11 = model(X_tensor).cpu().numpy().flatten()

# ======================
# Plot Curves
# ======================
plt.figure(figsize=(10,5))

plt.plot(freqs / 1e9, true_s11, label="True S11", linewidth=2)
plt.plot(freqs / 1e9, pred_s11, label="Predicted S11", linestyle="--")

plt.title(f"S11 Curve Comparison (Antenna {antenna_id})")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.legend()
plt.grid(True)

plt.show()
