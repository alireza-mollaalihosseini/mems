import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters
T = 20.0                      # Total simulation time (seconds)
t_rec = 3.0                   # Recording time (seconds)
noise = 1e-8                  # Noise magnitude
u_max = 1.0

a1 = np.arange(-2.0, -1.9, 0.1)
a2 = np.arange(-1.9, -1.8, 0.0005)
a3 = np.arange(-1.8, 3.5, 0.1)
a4 = np.arange(3.5, 3.7, 0.0005)
a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

a_full = np.concatenate([a1, a2, a3, a4, a5])

# Use tolerance-based matching
target_values = np.array([-1.8875, -1.8725, 3.5775, 3.5875])
tol = 1e-8
mask = np.any(np.abs(a_full[:, None] - target_values[None, :]) < tol, axis=1)

a_values = a_full[mask]
# print(f'the a_values are as follows : {a_values}')

for i, a in enumerate(a_values):

    # print(f'Processing a = {a:.2f} with i = {i}')
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))

    # Load results
    results_up_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more.npy')
    results_down_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more.npy')

    results_up   = results_up_full[mask, :]
    results_down = results_down_full[::-1, :][mask, :]

    data_up = results_up[i, :]
    data_down = results_down[i, :]

    t = np.linspace(T - t_rec, T, len(data_up))
    mask_window = (t >= 18.0) & (t <= 18.02)

    # Apply the mask to zoom in
    t_zoom = t[mask_window]
    data_up_zoom = data_up[mask_window]
    data_down_zoom = data_down[mask_window]

    ax.plot(t_zoom, data_up_zoom, label=f'Up, a={a:.5f}')
    ax.plot(t_zoom, data_down_zoom, label=f'Down, a={a:.5f}')

    ax.set_title(f'Time Series for a = {a:.2f}', fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Deflection (m)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'timeseries-a-{a:.5f}.png', bbox_inches='tight')
    plt.close()
