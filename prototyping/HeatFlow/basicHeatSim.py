import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_heat_heterogeneous(
    T_init,
    alpha_map,
    total_time=10.0,
    plate_size=1.0,
    visualize=True
):
    T = T_init.copy()
    N = T.shape[0]

    dx = plate_size / N
    alpha_max = np.max(alpha_map)

    # Stability-limited timestep
    dt = (dx * dx) / (4 * alpha_max) * 0.4
    total_steps = int(total_time / dt)

    print(f"Grid: {N} x {N}")
    print(f"dx = {dx:.6e}")
    print(f"dt = {dt:.6e}")
    print(f"Steps = {total_steps}")
    print(f"Initial mean temperature = {np.mean(T):.6f}")

    # Visualization setup
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax_alpha, ax_temp = axes

        alpha_img = ax_alpha.imshow(alpha_map, cmap="viridis")
        ax_alpha.set_title("Thermal Diffusivity Map α(x,y)")
        plt.colorbar(alpha_img, ax=ax_alpha)

        temp_img = ax_temp.imshow(T, cmap="hot")
        ax_temp.set_title("Temperature Field T(x,y)")
        plt.colorbar(temp_img, ax=ax_temp)

    # Progress bar
    pbar = tqdm(total=total_steps, desc="Simulating heat", unit="step")

    t = 0.0
    step = 0

    while step < total_steps:
        T_new = T.copy()

        # Face-centered diffusivity (averaged)
        axp = 0.5 * (alpha_map[1:-1, 1:-1] + alpha_map[2:, 1:-1])
        axm = 0.5 * (alpha_map[1:-1, 1:-1] + alpha_map[:-2, 1:-1])
        ayp = 0.5 * (alpha_map[1:-1, 1:-1] + alpha_map[1:-1, 2:])
        aym = 0.5 * (alpha_map[1:-1, 1:-1] + alpha_map[1:-1, :-2])

        # Flux-based heterogeneous heat update
        T_new[1:-1, 1:-1] = (
            T[1:-1, 1:-1] +
            dt / (dx * dx) * (
                axp * (T[2:, 1:-1] - T[1:-1, 1:-1]) -
                axm * (T[1:-1, 1:-1] - T[:-2, 1:-1]) +
                ayp * (T[1:-1, 2:] - T[1:-1, 1:-1]) -
                aym * (T[1:-1, 1:-1] - T[1:-1, :-2])
            )
        )

        # Neumann (insulated) boundaries
        T_new[0, :]  = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0]  = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]

        T = T_new

        # Visualization update
        if visualize and step % 5 == 0:
            temp_img.set_data(T)
            ax_temp.set_title(f"T(x,y) at t = {t:.3f}s")
            plt.pause(0.001)

        # tqdm update
        pbar.update(1)
        t += dt
        step += 1

    pbar.close()

    print(f"Final mean temperature = {np.mean(T):.6f}")

    if visualize:
        plt.show()

    return T, dt


# ============================
# Example Run
# ============================

if __name__ == "__main__":
    N = 128

    # Initial temperature field
    T_init = np.zeros((N, N))

    # Hot patch
    T_init[N//4:N//4+36, N//4:N//4+12] = 5000.0

    # Build heterogeneous alpha map
    alpha_map = np.ones((N, N)) * 9.7e-5  # base material

    # Fast-conducting center square
    alpha_map[N//3:2*N//3, N//3:2*N//3] = 6e-4

    # Insulating corner region
    alpha_map[:N//5, :N//5] = 1e-6

    # Run simulation
    final_T, dt = simulate_heat_heterogeneous(
        T_init,
        alpha_map,
        total_time=150.0,
        plate_size=1.0,
        visualize=True
    )
