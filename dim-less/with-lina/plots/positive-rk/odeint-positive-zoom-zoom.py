import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------- Parameters --------------------
T       = 500.0     # Total simulation time (seconds)
t_rec   = 3.0      # Recording time (seconds)
noise   = 1e-10     # Noise magnitude
u_max   = 1.0      # Max voltage (for filename variations)

# -------------------- Full a range and mask --------------------

a_full = np.linspace( 3.5825,  3.585, 200)


# a_full = np.arange(-2.0, 4.01, 0.01)
# Choose your zoom window here:
# mask = (a_full > 3.5) & (a_full < 3.7)      # zoom positive
mask = (a_full > 3.5825) & (a_full < 3.585)     # zoom zoom positive
# mask = (a_full > -1.9) & (a_full < -1.8)    # zoom negative
# mask = (a_full > -1.89) & (a_full < -1.87)  # zoom zoom negative
# mask = (a_full > -1.8775) & (a_full < -1.875)  # zoom zoom negative 1
# mask = (a_full > -1.8762) & (a_full < -1.8760)  # zoom zoom negative 1

# Cropped a_values
a_values = a_full[mask]

# -------------------- Load full results --------------------
results_up_full   = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-positive-side.npy')
results_down_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-positive-side.npy')
# results_down_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200.npy')

# try:
#     # Load the results
#     results_up_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-RK4-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200.npy')

# except FileNotFoundError:
#     print("Error: The specified result files could not be found.")
#     # filled resultss_up with nans similar to results_down shape
#     results_up_full = np.full(np.shape(results_down_full), np.nan)

# -------------------- Apply mask & reversal --------------------
# Upward: slice directly
results_up   = results_up_full[mask, :]            # shape (N, timesteps)
# Downward: reverse full, then slice
results_down = results_down_full[::-1, :][mask, :]  # shape (N, timesteps)

# -------------------- Storage for extrema --------------------
def init_lists():
    return [], []

# Up-sweep lists
a_peaks_up, mag_peaks_up         = init_lists()
a_valleys_up, mag_valleys_up     = init_lists()
a_global_max_up, mag_global_max_up = init_lists()
a_global_min_up, mag_global_min_up = init_lists()

# Down-sweep lists
a_peaks_down, mag_peaks_down         = init_lists()
a_valleys_down, mag_valleys_down     = init_lists()
a_global_max_down, mag_global_max_down = init_lists()
a_global_min_down, mag_global_min_down = init_lists()

# -------------------- Helper to extract & store extrema --------------------
def process_series(a, data,
                   peaks_lists, valleys_lists,
                   max_lists, min_lists):
    peaks, _   = find_peaks(data)
    valleys, _ = find_peaks(-data)

    # all local peaks/valleys
    for idx in peaks:
        peaks_lists[0].append(a)
        peaks_lists[1].append(data[idx])
    for idx in valleys:
        valleys_lists[0].append(a)
        valleys_lists[1].append(data[idx])

    # global max
    if peaks.size > 0:
        idx_max = peaks[np.argmax(data[peaks])]
        max_lists[0].append(a)
        max_lists[1].append(data[idx_max])
    elif np.std(data) < 1e-10:
        # constant series
        max_lists[0].append(a)
        max_lists[1].append(data[0])
    else:
        max_lists[0].append(a)
        max_lists[1].append(np.nan)

    # global min
    if valleys.size > 0:
        idx_min = valleys[np.argmin(data[valleys])]
        min_lists[0].append(a)
        min_lists[1].append(data[idx_min])
    elif np.std(data) < 1e-10:
        # constant series
        min_lists[0].append(a)
        min_lists[1].append(data[0])
    else:
        min_lists[0].append(a)
        min_lists[1].append(np.nan)

# -------------------- Loop over cropped range --------------------
for i, a in enumerate(a_values):
    data_up   = results_up[i, :]
    process_series(a, data_up,
                   [a_peaks_up, mag_peaks_up],
                   [a_valleys_up, mag_valleys_up],
                   [a_global_max_up, mag_global_max_up],
                   [a_global_min_up, mag_global_min_up])

    data_down = results_down[i, :]
    process_series(a, data_down,
                   [a_peaks_down, mag_peaks_down],
                   [a_valleys_down, mag_valleys_down],
                   [a_global_max_down, mag_global_max_down],
                   [a_global_min_down, mag_global_min_down])

# -------------------- Plotting --------------------
fig, ax = plt.subplots(figsize=(16, 8))

# background: all local extrema in grey
ax.scatter(a_peaks_up,   mag_peaks_up,   s=10, marker='x', color='grey', alpha=0.3, label='Local Peaks/Valleys')
ax.scatter(a_valleys_up, mag_valleys_up, s=10, marker='o', color='grey', alpha=0.3)
ax.scatter(a_peaks_down, mag_peaks_down, s=10, marker='x', color='grey', alpha=0.3)
ax.scatter(a_valleys_down, mag_valleys_down, s=10, marker='o', color='grey', alpha=0.3)

# foreground: global extrema
ax.scatter(a_global_max_up, mag_global_max_up, s=20, marker='x', color='red',      label='Upward Maxima')
ax.scatter(a_global_min_up, mag_global_min_up, s=20, marker='o', color='darkred',  label='Upward Minima')
ax.scatter(a_global_max_down, mag_global_max_down, s=20, marker='x', color='blue',     alpha=0.7, label='Downward Maxima')
ax.scatter(a_global_min_down, mag_global_min_down, s=20, marker='o', color='darkblue', alpha=0.7, label='Downward Minima')

# optional analytical vertical line
ax.axvline(x=3.583749685, color='black', linestyle='--', linewidth=2,
           label='a = 3.583749685 (analytical)')
# ax.axvline(x=-1.8760723518156315, color='black', linestyle='--', linewidth=2,
#            label='a = -1.8760723518156315 (analytical)')

# Formatting
ax.set_xlabel('a', fontsize=40, fontweight='bold')
ax.set_ylabel('Deflection (m)', fontsize=40, fontweight='bold')
# ax.set_ylim(3.14295e-7, 3.1431e-7)
# ax.set_ylim(3.1428e-7, 3.1432e-7)
# ax.set_ylim(1e-7, 8e-7)
ax.tick_params(axis='both', labelsize=20)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=20, loc='center left')
ax.margins(0)
ax.set_autoscale_on(False)

# Save figure
plt.tight_layout()
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-positive-zoom.png'
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-positive-zoom-zoom.png'
outname = f'extrema-RK4-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-positive-zoom-zoom.png'
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-negative-zoom.png'
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-zoom-zoom.png'
# outname = f'extrema-RK4-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-zoom-zoom.png'
# outname = f'extrema-RK4-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-zoom-zoom-zoom.png'
# outname = f'extrema-RK4-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-200-negative-zoom-zoom-zoom-close-to-fix-point.png'
plt.savefig(outname, bbox_inches='tight')
plt.close()
