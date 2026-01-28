import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

CSV_PATH = "../antenna_dataset_short.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DIPS = 15
EPOCHS = 130
LR = 1e-3

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

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)

# ============================
# LORENTZIAN PHYSICS
# ============================

def lorentzian_sum(freqs, amp, frq, gma):
    f = freqs.unsqueeze(2)
    frq = frq.unsqueeze(1)
    amp = amp.unsqueeze(1)
    gma = gma.unsqueeze(1)

    dips = -amp * (gma**2) / (gma**2 + (f - frq)**2)
    return dips.sum(dim=2)

# ============================
# HYBRID MODEL
# ============================

class HybridLorentzianRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (physics parameter predictor)
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

        # Residual RNN
        self.rnn = nn.GRU(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.rnn_out = nn.Linear(128, 1)

        self.softplus = nn.Softplus()

    def forward(self, s11, freqs):
        latent = self.encoder(s11)

        amp = torch.relu(self.amp_head(latent)) * 5.0 + 0.5

        f_min = freqs.min(dim=1, keepdim=True)[0]
        f_max = freqs.max(dim=1, keepdim=True)[0]
        frq_raw = self.frq_head(latent)
        frq = f_min + torch.sigmoid(frq_raw) * (f_max - f_min)

        gma = 5e7 + self.softplus(self.gma_head(latent)) * 3e8

        s11_physics = lorentzian_sum(freqs, amp, frq, gma)

        # Residual RNN correction
        rnn_input = s11_physics.unsqueeze(-1)
        rnn_out, _ = self.rnn(rnn_input)
        raw_residual = self.rnn_out(rnn_out).squeeze(-1)
        residual = torch.tanh(raw_residual) * 1.0  # max ±1 dB correction

        s11_final = s11_physics + residual

        return s11_final, s11_physics, residual, amp, frq, gma

model = HybridLorentzianRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================
# LOSS FUNCTION
# ============================

def loss_function(
    s11_true,
    s11_final,
    s11_physics,
    residual,
    amp,
    model,
    lambda_sparse=0.001,
    lambda_l2=0.0001,
    lambda_amp_floor=0.01,
    lambda_dip_match=0.6,
    lambda_residual=0.25,
    amp_floor=0.15
):
    # ========================
    # FINAL OUTPUT FIT LOSS
    # ========================
    
    
    #mse_final = torch.mean((s11_true - s11_final) ** 2)
    # Emphasize deep dips (more negative = higher weight)
    dip_weight = torch.exp(-s11_true / 5.0)

    mse_final = torch.mean(dip_weight * (s11_true - s11_final) ** 2)


    # ========================
    # LORENTZIAN PHYSICS LOSS
    # ========================
    sparsity_loss = torch.mean(torch.abs(amp))

    l2_loss = sum(torch.sum(p**2) for p in model.parameters())

    amp_floor_loss = torch.mean(torch.relu(amp_floor - amp))

    true_min = torch.min(s11_true)
    pred_min = torch.min(s11_physics)

    dip_match_loss = torch.relu(pred_min - true_min)

    # ========================
    # RESIDUAL REGULARIZATION
    # ========================
    residual_penalty = torch.mean(residual ** 2)

    smoothness_loss = torch.mean((residual[:,1:] - residual[:,:-1])**2)


    # ========================
    # TOTAL LOSS
    # ========================
    total_loss = (
        mse_final
        + lambda_sparse * sparsity_loss
        + lambda_l2 * l2_loss
        + lambda_amp_floor * amp_floor_loss
        + lambda_dip_match * dip_match_loss
        + lambda_residual * residual_penalty
        + 0.1 * smoothness_loss
    )

    return total_loss

# ============================
# TRAINING LOOP
# ============================

best_test_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for freqs, s11 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        freqs, s11 = freqs.to(DEVICE), s11.to(DEVICE)

        optimizer.zero_grad()

        s11_final, s11_phys, residual, amp, frq, gma = model(s11, freqs)

        loss = loss_function(s11, s11_final, s11_phys, residual, amp, model)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for freqs, s11 in test_loader:
            freqs, s11 = freqs.to(DEVICE), s11.to(DEVICE)
            s11_final, s11_phys, residual, amp, frq, gma = model(s11, freqs)
            loss = loss_function(s11, s11_final, s11_phys, residual, amp, model)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(f"Epoch {epoch+1} | Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "hybrid_lorentzian_rnn_best.pth")
        print("✅ Saved Best Hybrid Model")

# ============================
# SAVE FINAL MODEL
# ============================

torch.save(model.state_dict(), "hybrid_lorentzian_rnn_final.pth")
