import numpy as np
import matplotlib.pyplot as plt

# ============================
# Function Definition
# ============================
def lorentzian_sum(x, amp, frq, gma, n):
    amp = np.asarray(amp)
    frq = np.asarray(frq)
    gma = np.asarray(gma)

    y = np.zeros_like(x, dtype=float)

    for a, f, g in zip(amp, frq, gma):
        y += -a * g**2 / (g**2 + (x - f)**2)

    return y + n

if __name__ == "__main__":
    # ============================
    # Configuration
    # ============================
    NUM_TERMS = 15      # number of Lorentzian dips
    X_MIN = -10
    X_MAX = 10
    NUM_POINTS = 2000


    # ============================
    # Generate Random Parameters
    # ============================
    ampli = np.random.uniform(0, 25.0, NUM_TERMS)      # dip depths
    frequ = np.random.uniform(X_MIN, X_MAX, NUM_TERMS)  # center frequencies
    gamma = np.random.uniform(0.1, 1, NUM_TERMS)        # sharpness / Q-like factor
    n = 0.0                                            # vertical offset

    # ============================
    # Compute Curve
    # ============================
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
    y = lorentzian_sum(x, ampli, frequ, gamma, n)

    # ============================
    # Plot
    # ============================
    plt.figure(figsize=(12,6))
    plt.plot(x, y, linewidth=2)
    plt.title(f"Generalized Lorentzian Sum with {NUM_TERMS} Dips")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()
