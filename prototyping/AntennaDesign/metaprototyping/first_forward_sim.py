import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# USER-DEFINED ANTENNA
# -----------------------------

# Wavelength (normalised)
lam = 1.0
beta = 2 * np.pi / lam

# Antenna geometry (NxN planar array)
N = 4
d = 0.5 * lam

x = np.arange(N) * d
y = np.arange(N) * d
xx, yy = np.meshgrid(x, y)
positions = np.column_stack((xx.flatten(), yy.flatten()))

J = positions.shape[0]

# Excitations (user-defined)
amplitude = np.ones(J)
phase = np.zeros(J)  # radians
I = amplitude * np.exp(1j * phase)

# -----------------------------
# FORWARD FUNCTION (Equation 1)
# -----------------------------

theta = np.linspace(0, np.pi / 2, 120)
phi = np.linspace(0, 2 * np.pi, 240)
theta_grid, phi_grid = np.meshgrid(theta, phi)

FF = np.zeros_like(theta_grid, dtype=complex)

for i in range(J):
    xi, yi = positions[i]
    FF += I[i] * np.exp(
        -1j * beta * (
            xi * np.sin(theta_grid) * np.cos(phi_grid) +
            yi * np.sin(theta_grid) * np.sin(phi_grid)
        )
    )

# Radiation pattern magnitude
FF_mag = np.abs(FF)
FF_mag /= FF_mag.max()

# Convert to Cartesian for 3D plot
X = FF_mag * np.sin(theta_grid) * np.cos(phi_grid)
Y = FF_mag * np.sin(theta_grid) * np.sin(phi_grid)
Z = FF_mag * np.cos(theta_grid)

# -----------------------------
# PLOTS
# -----------------------------

fig = plt.figure(figsize=(14, 6))

# ---- Antenna geometry ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(positions[:, 0], positions[:, 1], np.zeros(J), s=40)
ax1.set_title("Antenna Array Geometry")
ax1.set_xlabel("x (λ)")
ax1.set_ylabel("y (λ)")
ax1.set_zlabel("z")

# ---- Radiation pattern ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(
    X, Y, Z,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)
ax2.set_title("3D Far-Field Radiation Pattern")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()
