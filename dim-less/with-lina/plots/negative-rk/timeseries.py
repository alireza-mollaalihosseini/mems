import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters
T       = 500.0     # Total simulation time (seconds)
t_rec   = 3.0      # Recording time (seconds)
noise   = 1e-9     # Noise magnitude
u_max   = 1.0      # Max voltage (for filename variations)

extra1 = np.linspace(-1.8775, -1.875, 50)
extra2 = np.linspace(-1.8762, -1.8760, 200)

# Combine all and sort
a_full = np.concatenate([extra1, extra2])
a_full = np.unique(np.round(a_full, decimals=8))

# Use tolerance-based matching
target_values = np.array([-1.876075])
tol = 1e-6
mask = np.any(np.abs(a_full[:, None] - target_values[None, :]) < tol, axis=1)

a_values = a_full[mask]
# print(f'the a_values are as follows : {a_values}')

for i, a in enumerate(a_values):

    # print(f'Processing a = {a:.2f} with i = {i}')
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))

    # Load results
    results_up_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-side.npy')
    results_down_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-side.npy')

    results_up   = results_up_full[mask, :]
    results_down = results_down_full[::-1, :][mask, :]

    data_up = results_up[i, :]
    data_down = results_down[i, :]

    t = np.linspace(T - t_rec, T, len(data_up))
    # mask_window = (t >= 18.0) & (t <= 18.02)

    # # Apply the mask to zoom in
    # t_zoom = t[mask_window]
    # data_up_zoom = data_up[mask_window]
    # data_down_zoom = data_down[mask_window]

    # ax.plot(t_zoom, data_up_zoom, label=f'Up, a={a:.5f}')
    # ax.plot(t_zoom, data_down_zoom, label=f'Down, a={a:.5f}')

    ax.plot(t[::100], data_up[::100], label=f'Up, a={a:.5f}', color='red')
    ax.plot(t[::100], data_down[::100], label=f'Down, a={a:.5f}', color='blue')


    ax.set_title(f'Time Series for a = {a:.2f}', fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Deflection (m)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'timeseries-a-{a}.png', bbox_inches='tight')
    plt.close()
