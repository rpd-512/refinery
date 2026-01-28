import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANTS
# ==========================================================
c0 = 3e8
mu0 = 4e-7 * np.pi
eps0 = 1 / (mu0 * c0**2)
eta0 = 50.0  # reference impedance

# ==========================================================
# ANTENNA (12x12)
# ==========================================================
antenna = np.array([
    [1,1,0,0,1,1,1,1,0,1,1,1],
    [1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,0,1,1,1,1],
    [1,1,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,1,0,0,0,0,1],
    [1,1,1,0,1,1,1,1,1,0,0,1]
], dtype=np.uint8)


feed_pixel = (0, 6)

pixel_size = 0.625e-3  # meters

# ==========================================================
# BUILD PIXEL COORDINATES
# ==========================================================
coords = []
for i in range(12):
    for j in range(12):
        if antenna[i, j]:
            x = (i - 6 + 0.5) * pixel_size
            y = (j - 6 + 0.5) * pixel_size
            coords.append((x, y))

coords = np.array(coords)
N = len(coords)

# Feed index
fx = (feed_pixel[0] - 6 + 0.5) * pixel_size
fy = (feed_pixel[1] - 6 + 0.5) * pixel_size
feed_idx = np.argmin(np.linalg.norm(coords - [fx, fy], axis=1))

# ==========================================================
# FREQUENCY SWEEP
# ==========================================================
freqs = np.linspace(10e9, 20e9, 81)
S11 = []

# ==========================================================
# MoM SOLVER LOOP
# ==========================================================


for f in freqs:
    k = 2 * np.pi * f / c0
    omega = 2 * np.pi * f

    Z = np.zeros((N, N), dtype=complex)

    for m in range(N):
        xm, ym = coords[m]
        for n in range(N):
            xn, yn = coords[n]
            r = np.sqrt((xm - xn)**2 + (ym - yn)**2)

            if m == n:
                # Self term (approximate)
                Z[m, n] = eta0 * (k * pixel_size / (2*np.pi)) * (1 - 1j)
            else:
                # Mutual impedance (Green's function)
                G = np.exp(-1j * k * r) / (4 * np.pi * r)
                Z[m, n] = 1j * omega * mu0 * G * pixel_size**2

    # Excitation vector (delta-gap)
    V = np.zeros(N, dtype=complex)
    V[feed_idx] = 1.0

    # Solve currents
    I = np.linalg.solve(Z, V)

    # Input impedance
    Zin = V[feed_idx] / I[feed_idx]

    # S11
    s11 = abs((Zin - eta0) / (Zin + eta0))
    S11.append(s11)

S11 = np.array(S11)
S11_dB = 20 * np.log10(S11 + 1e-12)

# ==========================================================
# PLOT
# ==========================================================
plt.plot(freqs / 1e9, S11_dB)
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
#plt.ylim(-40, 5)
plt.grid(True)
plt.show()
