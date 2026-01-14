import sys
sys.path.append("../../py/lib")

import py_refinery as rf
import numpy as np
import csv
import tqdm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn

# ======================== CONFIG ========================

filename = "kuka_youbot.csv"
target_vector = [150, 10, 200]
num_iterations = 500
data_count = 100

# ======================== UTILS ==========================

def fk_numpy(joint_angles):
    dh_table = np.array([
        [0, 147, 33, np.deg2rad(90)],
        [0,   0, 155, np.deg2rad(0)],
        [0,   0, 135, np.deg2rad(0)],
        [0,   0,   0, np.deg2rad(90)],
        [0, 217.5, 0, np.deg2rad(0)]
    ])
    T = np.eye(4)
    for i in range(len(dh_table)):
        theta = dh_table[i][0] + joint_angles[i]
        d = dh_table[i][1]
        a = dh_table[i][2]
        alpha = dh_table[i][3]

        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        T_i = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,      ca,     d],
            [0.0,   0.0,    0.0,   1.0]
        ])
        T = T @ T_i
    return T[:3, 3]

def l1_loss(y, y_hat):
    return nn.L1Loss()(torch.tensor(y_hat), torch.tensor(y)).item()

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
    features = [float(x) for x in row[5:8]]
    labels = [float(x) for x in row[8:13]]
    print(features, labels)
    dp.append(rf.Datapoint(6, features, labels))

# ======================== NEAREST NEIGHBOUR ENGINE =======

print("\nBuilding KD-tree index...")
engine = rf.NearestNeighbourEngine(3, [])
engine.insert_batch(dp)
nearest = engine.query(rf.Datapoint.from_vector(target_vector))

# ======================== OPTIMIZERS TO TEST =============

optimizers = {
    "GradientDescent": rf.GradientDescentOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.05),
    "GradientMomentum":rf.GradientMomentumOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.01),
    "GradientNesterov":rf.GradientNesterovOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.001, 0.7),
    "Adagrad":         rf.AdagradOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.015),
    "RMSprop":         rf.RMSpropOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.001, 0.9),
    "Adam":            rf.AdamOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.01)
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
    pos_refined = fk_numpy(refined)
    err = position_error(target_vector, pos_refined)

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
print("{:<20} {:<12} {:<12} {:<12}".format("Optimizer", "Time(s)", "Final Loss", "Pos Error"))
print("-" * 60)
for k, v in results.items():
    print(f"{k:<20} {v['time']:<12.4f} {v['final_loss']:<12.3f} {v['position_error']:<12.3f}")

plt.show()
