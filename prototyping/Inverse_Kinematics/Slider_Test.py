import sys
sys.path.append("../../py/lib")  # path to py-refinery compiled module

import py_refinery as rf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tqdm
import csv

# =====================
# Load dataset
# =====================
filename = "kuka_youbot.csv"
with open(filename, newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    data = [row for row in reader]


# =====================
# Forward Kinematics
# =====================
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


def fk_all_joints(joint_angles):
    """Returns list of (x,y,z) for each joint."""
    dh_table = np.array([
        [0, 147, 33, np.deg2rad(90)],
        [0,   0, 155, np.deg2rad(0)],
        [0,   0, 135, np.deg2rad(0)],
        [0,   0,   0, np.deg2rad(90)],
        [0, 217.5, 0, np.deg2rad(0)]
    ])

    T = np.eye(4)
    points = [T[:3, 3]]
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
        points.append(T[:3, 3])
    return np.array(points)


# =====================
# Setup Refinery Engine
# =====================
engine = rf.NearestNeighbourEngine(3, [])
dp = []
for row in tqdm.tqdm(data[:10]):
    features = [float(x) for x in row[5:8]]
    labels = [float(x) for x in row[8:13]]
    dp.append(rf.Datapoint(6, features, labels))
engine.insert_batch(dp)

adam = rf.AdamOptimizer(fk_numpy, rf.LossFunction.mse_loss, 0.01)
ref_engine = rf.RefinementEngine(adam)


# =====================
# Matplotlib Setup
# =====================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.2, bottom=0.25)

x0, y0, z0 = 450, 100, 100
ax.set_xlim(0, 600)
ax.set_ylim(-300, 300)
ax.set_zlim(0, 600)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("KUKA YouBot Inverse Kinematics Visualization")

# Initial robot + target
target_point = ax.scatter([x0], [y0], [z0], c='r', s=50, label='Target')
initial_joints = fk_all_joints([0, 0, 0, 0, 0])
robot_line, = ax.plot(initial_joints[:,0], initial_joints[:,1], initial_joints[:,2], '-o', lw=2, c='b', label='Robot')
ax.legend()


# =====================
# Sliders
# =====================
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_z = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

sx = Slider(ax_x, 'X', 0, 600, valinit=x0)
sy = Slider(ax_y, 'Y', -300, 300, valinit=y0)
sz = Slider(ax_z, 'Z', 0, 600, valinit=z0)


# =====================
# Update Function
# =====================
def update(val):
    target = [sx.val, sy.val, sz.val]

    # Query nearest datapoint and refine
    nearest = engine.query(rf.Datapoint.from_vector(target))
    ref_engine.set_seed(nearest)
    ref_engine.set_target(target)
    refined = ref_engine.refine(200)

    # Compute all joint positions
    joint_positions = fk_all_joints(refined)
    x, y, z = joint_positions[:,0], joint_positions[:,1], joint_positions[:,2]

    # Update robot
    robot_line.set_data(x, y)
    robot_line.set_3d_properties(z)

    # Update target marker
    target_point._offsets3d = ([sx.val], [sy.val], [sz.val])

    fig.canvas.draw_idle()


sx.on_changed(update)
sy.on_changed(update)
sz.on_changed(update)

plt.show()
