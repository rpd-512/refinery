import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# =========================================================
# PARAMETERS
# =========================================================

lam = 1.0
N = 32                  # aperture resolution
d = 0.5 * lam

# =========================================================
# BINARY APERTURE PLANE (0 = OFF, 1 = ON)
# =========================================================

aperture = np.zeros((N, N))
aperture[N//4:3*N//4, N//4:3*N//4] = 1

# =========================================================
# APERTURE GRID (FOR VISUALIZATION)
# =========================================================

x_edges = np.arange(N + 1) * d
y_edges = np.arange(N + 1) * d
Xg, Yg = np.meshgrid(x_edges, y_edges)
Zg = np.zeros_like(Xg)

colors = np.zeros((N, N, 4))
colors[aperture == 1] = [0.2, 0.8, 0.2, 1.0]   # ON
colors[aperture == 0] = [0.2, 0.4, 0.8, 1.0]   # OFF

# =========================================================
# FORWARD FUNCTION (FFT APERTURE MODEL)
# =========================================================

FF = np.fft.fftshift(np.fft.fft2(aperture))
FF_mag = np.abs(FF)
FF_mag /= FF_mag.max()

# =========================================================
# SMOOTHING (VISUAL ONLY)
# =========================================================

FF_mag = gaussian_filter(FF_mag, sigma=1.2)

# =========================================================
# ANGULAR COORDINATES (HIGHER RES FOR SMOOTHNESS)
# =========================================================

Nu = 6 * N
Nv = 6 * N

u = np.linspace(-1, 1, Nu)
v = np.linspace(-1, 1, Nv)
U, V = np.meshgrid(u, v)

# Interpolate FFT onto finer grid
from scipy.interpolate import RegularGridInterpolator

interp = RegularGridInterpolator(
    (np.linspace(-1, 1, N), np.linspace(-1, 1, N)),
    FF_mag,
    bounds_error=False,
    fill_value=0
)

pts = np.stack([U.ravel(), V.ravel()], axis=-1)
FF_mag_smooth = interp(pts).reshape(U.shape)

# Physical region
mask = U**2 + V**2 <= 1
FF_mag_smooth[~mask] = 0

# Convert to 3D
Z = FF_mag_smooth
X = FF_mag_smooth * U
Y = FF_mag_smooth * V

# =========================================================
# PLOTTING
# =========================================================

fig = plt.figure(figsize=(14, 6))

# ---- Aperture plane ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(
    Xg, Yg, Zg,
    facecolors=colors,
    edgecolor='k',
    linewidth=0.6,
    shade=False
)
ax1.set_title("Binary Aperture Plane")
ax1.set_xlabel("x (λ)")
ax1.set_ylabel("y (λ)")
ax1.set_zlabel("z")
ax1.set_zlim(-0.1, 0.1)

# ---- Radiation pattern ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(
    X, Y, Z,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)
ax2.set_title("3D Far-Field Radiation Pattern")
ax2.set_xlabel("u = sinθ cosφ")
ax2.set_ylabel("v = sinθ sinφ")
ax2.set_zlabel("|FF|")

plt.tight_layout()
plt.show()
