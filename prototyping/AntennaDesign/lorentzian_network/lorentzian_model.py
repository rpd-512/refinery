import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

import random

def plot_random_prediction(model, dataset, device):
    model.eval()

    idx = random.randint(0, len(dataset) - 1)
    freqs, s11 = dataset[idx]

    freqs = freqs.unsqueeze(0).to(device)
    s11 = s11.unsqueeze(0).to(device)

    with torch.no_grad():
        s11_pred, amp, frq, gma, baseline = model(s11, freqs)

    freqs = freqs.cpu().numpy()[0]
    s11 = s11.cpu().numpy()[0]
    s11_pred = s11_pred.cpu().numpy()[0]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e9, s11, label="Real S11", linewidth=2)
    plt.plot(freqs / 1e9, s11_pred, label="Predicted S11", linestyle="--")

    plt.title("Random Sample — Real vs Lorentzian Prediction")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================
# CONFIG
# ============================

CSV_PATH = "../antenna_dataset_short.csv"
NUM_DIPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
LR = 1e-4

# ============================
# UTILITIES
# ============================

def parse_array(text):
    """Parse string-stored arrays from CSV"""
    text = text.strip().strip("[]")
    values = re.split(r"[\s,]+", text)
    return np.array([float(v) for v in values if v])

# ============================
# LORENTZIAN PHYSICS MODEL
# ============================

def lorentzian_sum(freqs, amp, frq, gma, baseline):
    """
    freqs:     (B, N)
    amp:       (B, K)
    frq:       (B, K)
    gma:       (B, K)
    baseline:  (B, 1)
    """

    B, N = freqs.shape
    K = amp.shape[1]

    f = freqs.unsqueeze(2)
    frq = frq.unsqueeze(1)
    amp = amp.unsqueeze(1)
    gma = gma.unsqueeze(1)

    # Lorentzian dip equation
    dips = -amp * (gma**2) / (gma**2 + (f - frq)**2)

    s11 = dips.sum(dim=2)

    return s11 + baseline

# ============================
# DATASET LOADER
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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

NUM_POINTS = len(dataset[0][0])

# ============================
# NEURAL NETWORK
# ============================

class LorentzianNet(nn.Module):
    """
    Predicts 15 Lorentzian resonances from real S11
    """

    def __init__(self):
        super().__init__()

        # Encode S11 curve
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

        # Output heads
        self.amp_head = nn.Linear(256, NUM_DIPS)
        self.frq_head = nn.Linear(256, NUM_DIPS)
        self.gma_head = nn.Linear(256, NUM_DIPS)
        self.base_head = nn.Linear(256, 1)

        self.softplus = nn.Softplus()

    def forward(self, s11, freqs):
        latent = self.encoder(s11)

        # Dip depth (scaled to realistic dB)
        amp = torch.relu(self.amp_head(latent)) + 0.5

        # Force frequencies into RF band
        f_min = freqs.min(dim=1, keepdim=True)[0]
        f_max = freqs.max(dim=1, keepdim=True)[0]
        frq_raw = self.frq_head(latent)
        frq = f_min + torch.sigmoid(frq_raw) * (f_max - f_min)

        # Gamma / linewidth (RF realistic scale)
        gma_raw = self.gma_head(latent)
        gma = 5e7 + self.softplus(gma_raw) * 3e8

        # Baseline offset
        baseline = self.base_head(latent)

        # Physics forward model
        s11_pred = lorentzian_sum(freqs, amp, frq, gma, baseline)

        return s11_pred, amp, frq, gma, baseline


model = LorentzianNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================
# LOSS FUNCTION
# ============================

def loss_function(s11_true, s11_pred, amp, model,
                  lambda_sparse=0.001,
                  lambda_l2=0.0001,
                  lambda_amp_floor=0.01,
                  lambda_dip_match=0.15,
                  amp_floor=0.15):
    """
    Loss = Reconstruction MSE
         + Sparsity penalty
         + L2 regularization
         + Amplitude floor penalty
         + Max dip depth penalty (forces deep dips)
    """

    # Reconstruction loss
    mse = torch.mean((s11_true - s11_pred) ** 2)

    # Encourage unused dips to shrink
    sparsity_loss = torch.mean(torch.abs(amp)) if lambda_sparse > 0 else 0.0

    # Weight decay
    l2_loss = sum(torch.sum(p**2) for p in model.parameters()) if lambda_l2 > 0 else 0.0

    # Prevent amplitude collapse
    amp_floor_loss = torch.mean(torch.relu(amp_floor - amp))

    # ---- NEW: Max Dip Depth Penalty ----
    true_min = torch.min(s11_true)   # deepest true dip
    pred_min = torch.min(s11_pred)   # deepest predicted dip

    # Penalize if predicted dip is weaker (less negative)
    dip_match_loss = torch.relu(pred_min - true_min)

    # Final loss
    total_loss = (
        mse
        + lambda_sparse * sparsity_loss
        + lambda_l2 * l2_loss
        + lambda_amp_floor * amp_floor_loss
        + lambda_dip_match * dip_match_loss
    )

    return total_loss



# ============================
# TRAINING LOOP
# ============================

train_losses = []
test_losses = []

best_test_loss = float("inf")


for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    # -------- TRAIN --------
    for freqs, s11 in progress:
        freqs = freqs.to(DEVICE)
        s11 = s11.to(DEVICE)

        optimizer.zero_grad()

        s11_pred, amp, frq, gma, baseline = model(s11, freqs)

        loss = loss_function(s11, s11_pred, amp, model)

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        progress.set_postfix(train_loss=loss.item())

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -------- TEST --------
    model.eval()
    epoch_test_loss = 0

    with torch.no_grad():
        for freqs, s11 in test_loader:
            freqs = freqs.to(DEVICE)
            s11 = s11.to(DEVICE)

            s11_pred, amp, frq, gma, baseline = model(s11, freqs)

            loss = loss_function(s11, s11_pred, amp, model)

            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss = {avg_train_loss:.6f} | Test Loss = {avg_test_loss:.6f}")

    # Save best model checkpoint
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
        }, "best_lorentzian_model.pt")

        print("✅ Saved new best model")


    if epoch % 10 == 0:  # change to %5 or %10 if too frequent
        print("Predicted amp mean:", amp.mean().item())
        print("Predicted gamma mean:", gma.mean().item())
        print("Predicted freq range:", frq.min().item(), frq.max().item())
        print("Prediction std:", s11_pred.std().item())
        #plot_random_prediction(model, test_set, DEVICE)

# ============================
# VISUALIZE ONE SAMPLE FIT
# ============================

freqs, s11 = dataset[0]

freqs = freqs.unsqueeze(0).to(DEVICE)
s11 = s11.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    s11_pred, amp, frq, gma, baseline = model(s11, freqs)

freqs = freqs.cpu().numpy()[0]
s11 = s11.cpu().numpy()[0]
s11_pred = s11_pred.cpu().numpy()[0]

train_rmse = np.sqrt(train_losses[-1])
test_rmse = np.sqrt(test_losses[-1])

print("\n===== FINAL PERFORMANCE =====")
print(f"Train RMSE = {train_rmse:.3f} dB")
print(f"Test RMSE  = {test_rmse:.3f} dB")

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss (dB²)")
plt.title("Train vs Test Loss — Lorentzian Model")
plt.legend()
plt.grid(True)
plt.show()

#save the model
torch.save(model.state_dict(), "final_lorentzian_model.pth")