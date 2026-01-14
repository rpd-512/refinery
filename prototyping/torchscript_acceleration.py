import torch
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------
# Native Python / NumPy implementation
# ---------------------------------------
def dh_transform_np(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,      ca,       d],
        [0,        0,       0,       1]
    ], dtype=np.float32)

def fk_numpy(joint_angles, dh_table):
    T = np.eye(4, dtype=np.float32)
    for i in range(len(dh_table)):
        theta, d, a, alpha = dh_table[i]
        theta += joint_angles[i]
        T_i = dh_transform_np(theta, d, a, alpha)
        T = T @ T_i
    return T[:3, 3]

# ---------------------------------------
# PyTorch (TorchScript-safe) implementation
# ---------------------------------------
class DH_FK(torch.nn.Module):
    def __init__(self, dh_params):
        super().__init__()
        self.register_buffer('dh_params', torch.tensor(dh_params, dtype=torch.float32))

    def transform(self, theta, d, a, alpha):
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        T = torch.stack([
            torch.stack([ct, -st * ca, st * sa, a * ct]),
            torch.stack([st,  ct * ca, -ct * sa, a * st]),
            torch.stack([torch.tensor(0.0), sa, ca, d]),
            torch.tensor([0.0, 0.0, 0.0, 1.0])
        ])
        return T

    def forward(self, joint_angles):
        T = torch.eye(4)
        n = self.dh_params.shape[0]
        for i in range(n):
            p = self.dh_params[i]
            theta = p[0] + joint_angles[i]
            d = p[1]
            a = p[2]
            alpha = p[3]
            T_i = self.transform(theta, d, a, alpha)
            T = T @ T_i
        return T[:3, 3]

# ---------------------------------------
# Setup
# ---------------------------------------
dh_table = [
    [0, 0.3, 0.2, math.pi/2],
    [0, 0.0, 0.3, 0],
    [0, 0.0, 0.2, 0]
]
model = DH_FK(dh_table)
scripted_model = torch.jit.script(model)

joint_angles_np = np.array([0.2, 0.5, -0.3])
joint_angles_torch = torch.tensor([0.2, 0.5, -0.3])

# ---------------------------------------
# Benchmark
# ---------------------------------------
iters_list = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
numpy_times, normal_times, script_times = [], [], []

for iters in iters_list:
    # NumPy
    start = time.time()
    for _ in range(iters):
        _ = fk_numpy(joint_angles_np, dh_table)
    numpy_times.append(time.time() - start)

    # PyTorch
    start = time.time()
    for _ in range(iters):
        _ = model(joint_angles_torch)
    normal_times.append(time.time() - start)

    # TorchScript
    start = time.time()
    for _ in range(iters):
        _ = scripted_model(joint_angles_torch)
    script_times.append(time.time() - start)

# ---------------------------------------
# Plot
# ---------------------------------------
plt.figure(figsize=(7,5))
plt.plot(iters_list, numpy_times, '^-', label='Native Python (NumPy)')
plt.plot(iters_list, normal_times, 'o-', label='PyTorch')
plt.plot(iters_list, script_times, 's-', label='TorchScript')
plt.title("Forward Kinematics Runtime Comparison (DH Parameters)")
plt.xlabel("Iterations")
plt.ylabel("Total Runtime (seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------
# Compute average time per FK call
# ---------------------------------------
avg_numpy = [t / n for t, n in zip(numpy_times, iters_list)]
avg_torch = [t / n for t, n in zip(normal_times, iters_list)]
avg_script = [t / n for t, n in zip(script_times, iters_list)]

print("Average time per FK call (seconds):")
print(f"NumPy: {np.mean(avg_numpy):.8f}")
print(f"PyTorch: {np.mean(avg_torch):.8f}")
print(f"TorchScript: {np.mean(avg_script):.8f}")

# Optional: plot average per-call time instead of total
plt.figure(figsize=(7,5))
plt.plot(iters_list, avg_numpy, '^-', label='NumPy')
plt.plot(iters_list, avg_torch, 'o-', label='PyTorch')
plt.plot(iters_list, avg_script, 's-', label='TorchScript')
plt.title("Average Time per Forward Kinematics Call")
plt.xlabel("Iterations")
plt.ylabel("Time per Call (seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
