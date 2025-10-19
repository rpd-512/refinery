import sys
sys.path.append("lib")   # relative path from testing.py to the compiled module

import py_refinery as rf
import numpy as np
import csv
import tqdm
import matplotlib.pyplot as plt
import math
filename = "/home/rapidfire69/rapid/coding/researchWorks/inverse_kinematics/dataset_generation_and_compilation/kaggle_upload/kuka_youbot.csv"

with open(filename, newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # skip header if present
    data = [row for row in reader]


# Example usage

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

        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T_i = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,      ca,     d],
            [0.0,   0.0,    0.0,   1.0]
        ])
        T = T @ T_i
    return T[:3, 3]

engine = rf.NearestNeighbourEngine(3, [])

dp = []
for row in tqdm.tqdm(data[:]):
    features = [float(x) for x in row[5:8]]
    labels = [float(x) for x in row[8:13]]
    dp.append(rf.Datapoint(6, features, labels))

def my_loss(y, y_hat):
    sum = rf.LossFunction.mse_loss(y, y_hat)
    return math.sqrt(sum*len(y))

engine.insert_batch(dp)
search_vector = [450, -100, 200]
adam = rf.GradientNesterovOptimizer(fk_numpy, rf.LossFunction.euclidean_loss, 0.0001, 0.35)
ref_engine = rf.RefinementEngine(adam)

nearest = engine.query(rf.Datapoint.from_vector(search_vector))
ref_engine.set_seed(nearest)
ref_engine.set_logging(True)
ref_engine.set_target(search_vector)
refined = ref_engine.refine(10000)

print(fk_numpy(refined))

#plot history
history = ref_engine.get_loss_history()
losses = [h for h in history]

print(rf.LossFunction.mse_loss(fk_numpy(refined), search_vector))

plt.plot(losses)
#plt.yscale("log")
plt.xlabel("Iteration")
#plt.ylabel("Loss (log scale)")
plt.title("Refinement Loss over Iterations")
plt.grid()
plt.show()