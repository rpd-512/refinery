import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# PARAMETERS
# =========================================================

lam = 1.0                  # wavelength (normalized)
N = 256                    # aperture resolution (try 256 first, then 1024)
d = 0.5 * lam              # cell size (λ/2)

# =========================================================
# BINARY APERTURE (0 = OFF, 1 = ON)
# =========================================================

aperture = np.zeros((N, N))

# square ON region in the center
w = N // 3
c = N // 2
aperture[c-w:c+w, c-w:c+w] = 1

# =========================================================
# APODIZATION (THIS CREATES SMOOTH BEAMS)
# =========================================================

wx = np.hanning(N)
wy = np.hanning(N)
window = np.outer(wx, wy)

aperture = aperture * window

# =========================================================
# APERTURE GRID (FOR VISUALIZATION)
# =========================================================

x_edges = np.arange(N + 1) * d
y_edges = np.arange(N + 1) * d
Xg, Yg = np.meshgrid(x_edges, y_edges)
Zg = np.zeros_like(Xg)

# Colors: green = ON, blue = OFF
colors = np.zeros((N, N, 4))
colors[aperture > 0] = [0.2, 0.8, 0.2, 1.0]   # green
colors[aperture == 0] = [0.2, 0.4, 0.8, 1.0]  # blue

# =========================================================
# FORWARD FUNCTION (APERTURE → FAR FIELD)
# =========================================================

FF = np.fft.fftshift(np.fft.fft2(aperture))
FF_mag = np.abs(FF)

# Normalize
FF_mag /= FF_mag.max()

# dB scale (this is critical)
FF_db = 20 * np.log10(FF_mag + 1e-12)
FF_db -= FF_db.max()
FF_db[FF_db < -40] = -40

# =========================================================
# ANGULAR COORDINATES (u,v SPACE)
# =========================================================

u = np.linspace(-1, 1, N)
v = np.linspace(-1, 1, N)
U, V = np.meshgrid(u, v)

# Physical region only
mask = U**2 + V**2 <= 1
FF_db[~mask] = -40

# Convert to 3D Cartesian for plotting
Z = FF_db
X = U
Y = V

# =========================================================
# PLOTTING
# =========================================================

fig = plt.figure(figsize=(14, 6))

# -------- Binary aperture plane --------
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(
    Xg, Yg, Zg,
    facecolors=colors,
    edgecolor='k',
    linewidth=0.3,
    shade=False
)
ax1.set_title("Binary Aperture Plane (Apodized)")
ax1.set_xlabel("x (λ)")
ax1.set_ylabel("y (λ)")
ax1.set_zlabel("z")
ax1.set_zlim(-0.1, 0.1)

# -------- Far-field radiation pattern --------
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(
    X, Y, Z,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)

ax2.set_title("3D Far-Field Radiation Pattern (dB)")
ax2.set_xlabel("u = sinθ cosφ")
ax2.set_ylabel("v = sinθ sinφ")
ax2.set_zlabel("Magnitude (dB)")

# Zoom to beam region (VERY IMPORTANT)
lim = 0.02
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-40, 0)

plt.tight_layout()
plt.show()

# =========================================================
# OPTIONAL: 2D CUT (GROUND TRUTH VIEW)
# =========================================================

mid = N // 2
plt.figure()
plt.plot(u, FF_db[mid, :])
plt.xlabel("u = sinθ cosφ")
plt.ylabel("Magnitude (dB)")
plt.title("Far-Field Cut (v = 0)")
plt.grid()
plt.show()
