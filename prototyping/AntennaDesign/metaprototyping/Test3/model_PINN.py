import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

CSV_PATH = "antenna_long_format_fixed.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-3
TEST_SPLIT = 0.2

# ============================================================
# Dataset: Group rows into full sweeps per antenna
# ============================================================
class FullSweepDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        grouped = df.groupby("antenna_id")
        self.samples = []

        for _, g in grouped:
            g = g.sort_values("frequency")

            length = g["length"].iloc[0]
            width = g["width"].iloc[0]
            feed_y = g["feed_y"].iloc[0]

            freqs = g["frequency"].values / 1e9
            s11 = g["s11"].values

            geom = np.array([length, width, feed_y], dtype=np.float32)
            self.samples.append((geom, freqs, s11))

        self.freqs = torch.tensor(self.samples[0][1], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        geom, freqs, s11 = self.samples[idx]
        return (
            torch.tensor(geom, dtype=torch.float32),
            torch.tensor(s11, dtype=torch.float32)
        )

dataset = FullSweepDataset(CSV_PATH)

test_size = int(len(dataset) * TEST_SPLIT)
train_size = len(dataset) - test_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

FREQS = dataset.freqs.to(DEVICE)
NUM_FREQ = len(FREQS)

print(f"Samples: {len(dataset)} | Frequencies per sweep: {NUM_FREQ}")

# ============================================================
# Spectrum-Aware Neural Network (Smooth Decoder)
# ============================================================
class S11SpectrumNet(nn.Module):
    def __init__(self, num_freq):
        super().__init__()

        # Geometry encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Spectrum decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_freq)
        )

        # Spectral smoothing conv
        self.smooth = nn.Conv1d(1, 1, kernel_size=9, padding=4, bias=False)

        # Initialize smoothing kernel (moving average)
        with torch.no_grad():
            self.smooth.weight[:] = 1 / 9

    def forward(self, geom):
        latent = self.encoder(geom)
        s11 = self.decoder(latent)

        # Smooth spectrum
        s11 = self.smooth(s11.unsqueeze(1)).squeeze(1)
        return s11


model = S11SpectrumNet(NUM_FREQ).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()

train_losses = []
test_losses = []

# ============================================================
# Training Loop
# ============================================================
print("Training full-sweep smooth spectrum model...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for geom, s11 in pbar:
        geom = geom.to(DEVICE)
        s11 = s11.to(DEVICE)

        pred = model(geom)

        # Base MSE
        loss = mse(pred, s11)

        # Optional smoothness penalty (light)
        smooth_loss = mse(pred[:, 1:] - pred[:, :-1], s11[:, 1:] - s11[:, :-1])
        loss = loss + 0.01 * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ---- TEST ----
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for geom, s11 in test_loader:
            geom = geom.to(DEVICE)
            s11 = s11.to(DEVICE)

            pred = model(geom)

            loss = mse(pred, s11)
            smooth_loss = mse(pred[:, 1:] - pred[:, :-1], s11[:, 1:] - s11[:, :-1])
            loss = loss + 0.01 * smooth_loss

            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}: Train={train_loss:.6f} | Test={test_loss:.6f}")

# ============================================================
# Save Model
# ============================================================
torch.save(model.state_dict(), "full_sweep_smooth_model.pt")
print("Saved → full_sweep_smooth_model.pt")

# ============================================================
# Plot Loss Curve
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve — Smooth Spectrum Model")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Visual Check: True vs Predicted Spectrum
# ============================================================
model.eval()

geom, s11_true = test_set[np.random.randint(len(test_set))]
geom = geom.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    s11_pred = model(geom).cpu().numpy()[0]

s11_true = s11_true.numpy()
freqs = FREQS.cpu().numpy()

plt.figure(figsize=(9,5))
plt.plot(freqs, s11_true, label="True", linewidth=2)
plt.plot(freqs, s11_pred, "--", label="Predicted", linewidth=2)
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.title("S11 Curve Comparison — Smooth Spectrum Model")
plt.legend()
plt.grid(True)
plt.show()
