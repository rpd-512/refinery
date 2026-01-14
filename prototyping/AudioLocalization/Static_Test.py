import sys
sys.path.append("../../py/lib")  # path to py-refinery compiled module

import py_refinery as rf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tqdm
import csv
import time


# =========================
# CONFIGURATION
# =========================
N_SAMPLES = 5000       # number of source positions to generate
C = 343.0              # speed of sound in m/s
NOISE_STD_TDOA = 1e-5  # Gaussian noise for TDOAs (seconds)
NOISE_STD_AMP = 0.01   # Gaussian noise for amplitudes

# Mic positions (example: triangle in XY plane)
mic_positions = np.array([
    [0.0, 0.0, 0.0],      # mic 1
    [0.2, 0.0, 0.0],      # mic 2
    [0.1, 0.173, 0.0]     # mic 3
])

# Source position bounds (X, Y, Z)
source_bounds = {
    'x': (-1.0, 1.0),
    'y': (-1.0, 1.0),
    'z': (0.0, 1.0)
}

# ======================== CONFIG ========================

filename = "trig_3_mic.csv"
target_vector = [1,0.9,0.9,0.0004935,1.000472797,4.00582e-05]
num_iterations = 500
data_count = 100


# ======================== UTILS ==========================
def mic_numpy(source_pos, mic_positions=mic_positions, c=C):
    source_pos = np.asarray(source_pos, dtype=float)

    distances = np.linalg.norm(mic_positions - source_pos[None, :], axis=1)
    amps = 1 / (distances**2)

    tdoa_12 = (distances[1] - distances[0]) / c
    tdoa_13 = (distances[2] - distances[0]) / c
    tdoa_23 = (distances[2] - distances[1]) / c

    tdoas = np.array([tdoa_12, tdoa_13, tdoa_23])

    out = np.concatenate([amps, tdoas])
    return out

def position_error(true_pos, pred_pos):
    return np.linalg.norm(np.array(true_pos) - np.array(pred_pos))

# ======================== DATA LOAD ======================

print("Loading dataset...")
with open(filename, newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    data = [row for row in reader]

dp = []
for row in tqdm.tqdm(data[:data_count]):
    features = [float(x) for x in row[:6]]
    labels = [float(x) for x in row[6:]]
    dp.append(rf.Datapoint(5, features, labels))

# ======================== NEAREST NEIGHBOUR ENGINE =======

print("\nBuilding KD-tree index...")
engine = rf.NearestNeighbourEngine(6, [])
engine.insert_batch(dp)
nearest = engine.query(rf.Datapoint.from_vector(target_vector))

# ======================== OPTIMIZERS TO TEST =============

optimizers = {
    "GradientDescent": rf.GradientDescentOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.05),
    "GradientMomentum":rf.GradientMomentumOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.01),
    "GradientNesterov":rf.GradientNesterovOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.001, 0.7),
    "Adagrad":         rf.AdagradOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.015),
    "RMSprop":         rf.RMSpropOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.001, 0.9),
    "Adam":            rf.AdamOptimizer(mic_numpy, rf.LossFunction.mse_loss, 0.01)
}

# ======================== TEST LOOP ======================

results = {}
plt.figure(figsize=(8,5))
print("\nStarting refinement tests...\n")
for name, optimizer in optimizers.items():
    print(f"Testing {name}...")

    ref_engine = rf.RefinementEngine(optimizer)
    ref_engine.set_seed(nearest)
    ref_engine.set_target(target_vector)
    ref_engine.set_logging(True)

    start_time = time.time()
    refined = ref_engine.refine(num_iterations)
    elapsed = time.time() - start_time

    losses = ref_engine.get_loss_history()
    final_loss = losses[-1]
    dat_refined = mic_numpy(refined)
    err = position_error(target_vector, dat_refined)

    results[name] = {
        "time": elapsed,
        "final_loss": final_loss,
        "position_error": err
    }

    plt.plot(losses, label=name)

# ======================== RESULTS ========================

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Refinement Loss Comparison")
plt.legend()
plt.tight_layout()

print("\n=== Performance Summary ===")
print("{:<20} {:<12} {:<12}".format("Optimizer", "Time(s)", "Final Loss"))
print("-" * 60)
for k, v in results.items():
    print(f"{k:<20} {v['time']:<12.4f} {v['final_loss']:<12.3f}")

plt.show()
