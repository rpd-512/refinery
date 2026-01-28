import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = "s11_model_2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Model Definition (same as training)
# ======================
class S11Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# Load Model Once
# ======================
model = S11Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# Prediction Function
# ======================
def predict_s11(length, width, feed_y, frequency_hz):
    """
    Predict S11 given antenna geometry and frequency.
    frequency_hz should be in Hz.
    """

    freq_norm = frequency_hz / 1e9  # normalize like training

    X = np.array([length, width, feed_y, freq_norm], dtype=np.float32)
    X_tensor = torch.tensor(X).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(X_tensor).item()

    print(f"S11 Prediction:")
    print(f"  Length   = {length}")
    print(f"  Width    = {width}")
    print(f"  Feed_y   = {feed_y}")
    print(f"  Frequency= {frequency_hz/1e9:.4f} GHz")
    print(f"  Pred S11 = {pred:.6f}")

    return pred

import time

t0 = time.perf_counter()

p = predict_s11(
    length=40.0,
    width=48.0,
    feed_y=-12.0,
    frequency_hz=2.45e9
)

t1 = time.perf_counter()

print(f"\nPredicted S11 = {p:.6f}")
print(f"Inference Time = {(t1 - t0)*1000:.4f} ms")
