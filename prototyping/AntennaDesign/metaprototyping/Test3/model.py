import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

CSV_PATH = "antenna_long_format_fixed.csv"
BATCH_SIZE = 4096
EPOCHS = 250
LR = 1e-3
TEST_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Dataset Loader
# ======================
class AntennaDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        X = df[["length", "width", "feed_y", "frequency"]].values
        y = df["s11"].values

        # Normalize frequency to GHz
        X[:, 3] = X[:, 3] / 1e9

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = AntennaDataset(CSV_PATH)

# ======================
# Train/Test Split
# ======================
test_size = int(len(dataset) * TEST_SPLIT)
train_size = len(dataset) - test_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train samples: {train_size}")
print(f"Test samples: {test_size}")

# ======================
# Neural Network
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

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []

# ======================
# Training Loop
# ======================
print(f"Training on {DEVICE}...")

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")

    for X, y in pbar:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

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
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f}")

# ======================
# Save Model
# ======================
torch.save(model.state_dict(), "s11_model_3.pt")
print("Model saved -> s11_model_3.pt")

# ======================
# Plot Loss Curves
# ======================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Test Loss (S11 Predictor)")
plt.legend()
plt.grid(True)
plt.show()
