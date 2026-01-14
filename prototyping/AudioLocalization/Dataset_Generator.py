import numpy as np
import pandas as pd

# =========================
# CONFIGURATION
# =========================
N_SAMPLES = 5000       # number of source positions to generate
C = 343.0              # speed of sound in m/s
NOISE_STD_TDOA = 1e-5  # Gaussian noise for TDOAs (seconds)
NOISE_STD_AMP = 0.01   # Gaussian noise for amplitudes

# Mic positions (example: triangle in XY plane)
mic_positions = np.array([
    [0.0, 0.0, 0.0],      # mic 1
    [0.2, 0.0, 0.0],      # mic 2
    [0.1, 0.173, 0.0]     # mic 3
])

# Source position bounds (X, Y, Z)
source_bounds = {
    'x': (-1.0, 1.0),
    'y': (-1.0, 1.0),
    'z': (0.0, 1.0)
}

# =========================
# FORWARD FUNCTION
# =========================
def forward_function(source_pos, mic_positions=mic_positions, c=C):
    """
    Given a 3D source position, compute amplitudes and TDOAs for 3 microphones.
    
    Args:
        source_pos: np.array of shape (3,) -> [x, y, z]
        mic_positions: np.array of shape (3,3)
        c: speed of sound
    
    Returns:
        amps: np.array of shape (3,)
        tdoas: np.array of shape (3,) -> [tdoa_12, tdoa_13, tdoa_23]
    """
    # compute distances
    distances = np.linalg.norm(mic_positions - source_pos[None,:], axis=1)
    
    # amplitudes (inverse-square law)
    amps = 1 / (distances**2)
    
    # tdoas
    tdoa_12 = (distances[1] - distances[0]) / c
    tdoa_13 = (distances[2] - distances[0]) / c
    tdoa_23 = (distances[2] - distances[1]) / c
    
    tdoas = np.array([tdoa_12, tdoa_13, tdoa_23])
    
    return amps, tdoas

# =========================
# GENERATE RANDOM SOURCES
# =========================
source_positions = np.random.uniform(
    low=[source_bounds['x'][0], source_bounds['y'][0], source_bounds['z'][0]],
    high=[source_bounds['x'][1], source_bounds['y'][1], source_bounds['z'][1]],
    size=(N_SAMPLES, 3)
)

# =========================
# COMPUTE FEATURES USING FORWARD FUNCTION
# =========================
all_amps = []
all_tdoas = []

for pos in source_positions:
    amps, tdoas = forward_function(pos)
    
    # add optional Gaussian noise
    if NOISE_STD_AMP > 0:
        amps += np.random.normal(0, NOISE_STD_AMP, amps.shape)
    if NOISE_STD_TDOA > 0:
        tdoas += np.random.normal(0, NOISE_STD_TDOA, tdoas.shape)
    
    all_amps.append(amps)
    all_tdoas.append(tdoas)

all_amps = np.array(all_amps)
all_tdoas = np.array(all_tdoas)

# =========================
# SAVE TO CSV
# =========================
df = pd.DataFrame({
    'mic1_amp': all_amps[:,0],
    'mic2_amp': all_amps[:,1],
    'mic3_amp': all_amps[:,2],
    'tdoa_12': all_tdoas[:,0],
    'tdoa_13': all_tdoas[:,1],
    'tdoa_23': all_tdoas[:,2],
    'target_x': source_positions[:,0],
    'target_y': source_positions[:,1],
    'target_z': source_positions[:,2]
})

df.to_csv('synthetic_dataset.csv', index=False)
print(f"Generated synthetic dataset with {N_SAMPLES} samples → 'synthetic_dataset.csv'")
