import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import random

CSV_PATH = "../antenna_dataset_short.csv"
MODEL_PATH = "model_002/final_lorentzian_model.pth"
NUM_DIPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# UTILITIES
# ============================

def parse_array(text):
    text = text.strip().strip("[]")
    values = re.split(r"[\s,]+", text)
    return np.array([float(v) for v in values if v])

# ============================
# DATASET
# ============================

class AntennaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        freqs = parse_array(row["frequencies"])
        s11 = parse_array(row["s11"])

        order = np.argsort(freqs)
        freqs = freqs[order]
        s11 = s11[order]

        return (
            torch.tensor(freqs, dtype=torch.float32),
            torch.tensor(s11, dtype=torch.float32)
        )

dataset = AntennaDataset(CSV_PATH)
NUM_POINTS = len(dataset[0][0])

# ============================
# PHYSICS MODEL
# ============================

def lorentzian_sum(freqs, amp, frq, gma, baseline):
    B, N = freqs.shape
    K = amp.shape[1]

    f = freqs.unsqueeze(2)
    frq = frq.unsqueeze(1)
    amp = amp.unsqueeze(1)
    gma = gma.unsqueeze(1)

    dips = -amp * (gma**2) / (gma**2 + (f - frq)**2)
    s11 = dips.sum(dim=2)

    return s11 + baseline

# ============================
# NETWORK
# ============================

class LorentzianNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(NUM_POINTS, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.amp_head = nn.Linear(256, NUM_DIPS)
        self.frq_head = nn.Linear(256, NUM_DIPS)
        self.gma_head = nn.Linear(256, NUM_DIPS)
        self.base_head = nn.Linear(256, 1)

        self.softplus = nn.Softplus()

    def forward(self, s11, freqs):
        latent = self.encoder(s11)

        amp = torch.relu(self.amp_head(latent)) + 0.5

        f_min = freqs.min(dim=1, keepdim=True)[0]
        f_max = freqs.max(dim=1, keepdim=True)[0]
        frq = f_min + torch.sigmoid(self.frq_head(latent)) * (f_max - f_min)

        gma = 5e7 + self.softplus(self.gma_head(latent)) * 3e8

        baseline = self.base_head(latent)

        s11_pred = lorentzian_sum(freqs, amp, frq, gma, baseline)

        return s11_pred

# ============================
# LOAD MODEL
# ============================

model = LorentzianNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================
# RANDOM PLOT FUNCTION
# ============================

def plot_random_sample():
    idx = random.randint(0, len(dataset) - 1)
    freqs, s11_true = dataset[idx]

    freqs = freqs.unsqueeze(0).to(DEVICE)
    s11_true = s11_true.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        s11_pred = model(s11_true, freqs)

    freqs = freqs.cpu().numpy()[0]
    s11_true = s11_true.cpu().numpy()[0]
    s11_pred = s11_pred.cpu().numpy()[0]

    plt.figure(figsize=(10, 5))

    plt.plot(freqs / 1e9, s11_true, label="True S11", linewidth=2)
    plt.plot(freqs / 1e9, s11_pred, label="Predicted S11", linestyle="--")

    plt.title(f"Random Sample #{idx} — True vs Predicted S11")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# RUN RANDOM VISUALIZATION
# ============================

plot_random_sample()
