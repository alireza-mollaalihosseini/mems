# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # Parameters
# T = 20.0                      # Total simulation time (seconds)
# t_rec = 3.0                   # Recording time (seconds)
# noise = 1e-7                  # Noise magnitude
# u_max = 1.0

# # a1 = np.arange(-2.0, -1.9, 0.1)
# # a2 = np.arange(-1.9, -1.8, 0.0005)
# # a3 = np.arange(-1.8, 3.5, 0.1)
# # a4 = np.arange(3.5, 3.7, 0.0005)
# # a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# # a_values = np.concatenate([a1, a2, a3, a4, a5])

# # Original ranges
# a1 = np.arange(-2.0, -1.9, 0.1)
# a2 = np.arange(-1.9, -1.8, 0.0005)
# a3 = np.arange(-1.8, 3.5, 0.1)
# a4 = np.arange(3.5, 3.7, 0.0005)
# a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# # Additional 20 points between specified intervals
# extra1 = np.linspace(-1.8775, -1.875, 20)
# extra2 = np.linspace(3.8525, 3.855, 20)

# # Combine all and sort
# a_values = np.concatenate([a1, a2, extra1, a3, a4, extra2, a5])
# a_values = np.unique(np.round(a_values, decimals=8))

# # a_values = a_values[(a_values > 3.0)&(a_values < 4.0)] # positive zoom
# # a_values = a_values[(a_values > 3.58) & (a_values < 3.6)]  # positive zoom zoom
# # a_values = a_values[(a_values > -1.9) & (a_values < -1.8)]  # negative zoom
# # # reverse the order for negative zoom
# # a_values = a_values[::-1]

# # Load the results
# results_up = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20.npy')
# results_down = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20.npy')

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(16, 8))

# # Lists to store peaks and valleys
# a_peaks_up, mag_peaks_up = [], []
# a_valleys_up, mag_valleys_up = [], []
# a_peaks_down, mag_peaks_down = [], []
# a_valleys_down, mag_valleys_down = [], []
# a_global_max_up, mag_global_max_up = [], []
# a_global_min_up, mag_global_min_up = [], []
# a_global_max_down, mag_global_max_down = [], []
# a_global_min_down, mag_global_min_down = [], []

# # Process each a value
# for i, a in enumerate(a_values):
#     # Upward sweep: direct correspondence
#     data_up = results_up[i, :]
#     peaks_up, _ = find_peaks(data_up)
#     valleys_up, _ = find_peaks(-data_up)
    
#     # Store all peaks for upward sweep (for grey plotting)
#     for peak_idx in peaks_up:
#         a_peaks_up.append(a)
#         mag_peaks_up.append(data_up[peak_idx])
#     # Store all valleys for upward sweep (for grey plotting)
#     for valley_idx in valleys_up:
#         a_valleys_up.append(a)
#         mag_valleys_up.append(data_up[valley_idx])
    
#     # Handle global extrema for upward sweep
#     if len(peaks_up) > 0:
#         max_idx = peaks_up[np.argmax(data_up[peaks_up])]
#         a_global_max_up.append(a)
#         mag_global_max_up.append(data_up[max_idx])
#     if len(valleys_up) > 0:
#         min_idx = valleys_up[np.argmin(data_up[valleys_up])]
#         a_global_min_up.append(a)
#         mag_global_min_up.append(data_up[min_idx])
    
#     # Handle constant series or no peaks/valleys
#     if len(peaks_up) == 0 and len(valleys_up) == 0 and np.std(data_up) < 1e-10:
#         a_global_max_up.append(a)
#         mag_global_max_up.append(data_up[0])
#         a_global_min_up.append(a)
#         mag_global_min_up.append(data_up[0])
#         a_peaks_up.append(a)
#         mag_peaks_up.append(data_up[0])
#         a_valleys_up.append(a)
#         mag_valleys_up.append(data_up[0])
#     elif len(peaks_up) == 0:
#         a_global_max_up.append(a)
#         mag_global_max_up.append(np.nan)
#     elif len(valleys_up) == 0:
#         a_global_min_up.append(a)
#         mag_global_min_up.append(np.nan)
    
#     # Downward sweep: reversed index
#     j = len(a_values) - 1 - i
#     data_down = results_down[j, :]
#     peaks_down, _ = find_peaks(data_down)
#     valleys_down, _ = find_peaks(-data_down)
    
#     # Store all peaks for downward sweep (for grey plotting)
#     for peak_idx in peaks_down:
#         a_peaks_down.append(a)
#         mag_peaks_down.append(data_down[peak_idx])
#     # Store all valleys for downward sweep (for grey plotting)
#     for valley_idx in valleys_down:
#         a_valleys_down.append(a)
#         mag_valleys_down.append(data_down[valley_idx])
    
#     # Handle global extrema for downward sweep
#     if len(peaks_down) > 0:
#         max_idx = peaks_down[np.argmax(data_down[peaks_down])]
#         a_global_max_down.append(a)
#         mag_global_max_down.append(data_down[max_idx])
#     if len(valleys_down) > 0:
#         min_idx = valleys_down[np.argmin(data_down[valleys_down])]
#         a_global_min_down.append(a)
#         mag_global_min_down.append(data_down[min_idx])
    
#     # Handle constant series or no peaks/valleys
#     if len(peaks_down) == 0 and len(valleys_down) == 0 and np.std(data_down) < 1e-10:
#         a_global_max_down.append(a)
#         mag_global_max_down.append(data_down[0])
#         a_global_min_down.append(a)
#         mag_global_min_down.append(data_down[0])
#         a_peaks_down.append(a)
#         mag_peaks_down.append(data_down[0])
#         a_valleys_down.append(a)
#         mag_valleys_down.append(data_down[0])
#     elif len(peaks_down) == 0:
#         a_global_max_down.append(a)
#         mag_global_max_down.append(np.nan)
#     elif len(valleys_down) == 0:
#         a_global_min_down.append(a)
#         mag_global_min_down.append(np.nan)

# # Plot all local peaks and valleys in grey (background)
# ax.scatter(a_peaks_up, mag_peaks_up, s=10, marker='x', color='grey', alpha=0.3, label='Local Peaks/Valleys')
# ax.scatter(a_valleys_up, mag_valleys_up, s=10, marker='o', color='grey', alpha=0.3)
# ax.scatter(a_peaks_down, mag_peaks_down, s=10, marker='x', color='grey', alpha=0.3)
# ax.scatter(a_valleys_down, mag_valleys_down, s=10, marker='o', color='grey', alpha=0.3)

# # Plot global maxima and minima in original colors (foreground)
# ax.scatter(a_global_max_up, mag_global_max_up, s=20, marker='x', color='red', label='Upward Maxima')
# ax.scatter(a_global_min_up, mag_global_min_up, s=20, marker='o', color='darkred', label='Upward Minima')
# ax.scatter(a_global_max_down, mag_global_max_down, s=20, marker='x', color='blue', alpha=0.7, label='Downward Maxima')
# ax.scatter(a_global_min_down, mag_global_min_down, s=20, marker='o', color='darkblue', alpha=0.7, label='Downward Minima')

# plt.axvline(x=3.583749685, color='black', linestyle='--', linewidth=2, label='a = 3.583749685 (analytical)') # positive side
# plt.axvline(x=-1.876072391, color='black', linestyle='--', linewidth=2, label='a = -1.876072391 (analytical)') # negative side

# # Formatting the plot
# ax.set_xlabel('a', fontsize=40)
# ax.set_ylabel('Deflection (m)', fontsize=40)
# ax.margins(0)
# ax.set_autoscale_on(False)
# ax.tick_params(axis='both', labelsize=25)
# ax.xaxis.offsetText.set_fontsize(40)
# ax.yaxis.offsetText.set_fontsize(40)
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)
# ax.legend(fontsize=20, loc='upper center')

# # Save and close
# plt.tight_layout()
# plt.savefig(f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20.png', bbox_inches='tight')
# # plt.savefig(f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-positive-zoom.png', bbox_inches='tight')
# # plt.savefig(f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-positive-zoom-zoom.png', bbox_inches='tight')
# plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------- Parameters --------------------
T       = 50.0     # Total simulation time (seconds)
t_rec   = 3.0      # Recording time (seconds)
noise   = 1e-8     # Noise magnitude
u_max   = 1.0      # Max voltage (for filename variations)

# -------------------- Full a range and mask --------------------
# a1 = np.arange(-2.0, -1.9, 0.1)
# a2 = np.arange(-1.9, -1.8, 0.0005)
# a3 = np.arange(-1.8, 3.5, 0.1)
# a4 = np.arange(3.5, 3.7, 0.0005)
# a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# a_full = np.concatenate([a1, a2, a3, a4, a5])

# Original ranges
a1 = np.arange(-2.0, -1.9, 0.1)
a2 = np.arange(-1.9, -1.8, 0.0005)
a3 = np.arange(-1.8, 3.5, 0.1)
a4 = np.arange(3.5, 3.7, 0.0005)
a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# Additional 20 points between specified intervals
extra1 = np.linspace(-1.8775, -1.875, 20)
extra2 = np.linspace(3.8525, 3.855, 20)

# Combine all and sort
a_full = np.concatenate([a1, a2, extra1, a3, a4, extra2, a5])
a_full = np.unique(np.round(a_full, decimals=8))


# a_full = np.arange(-2.0, 4.01, 0.01)
# Choose your zoom window here:
# mask = (a_full > 3.5) & (a_full < 3.7)      # zoom positive
# mask = (a_full > 3.575) & (a_full < 3.595)     # zoom zoom positive
# mask = (a_full > -1.9) & (a_full < -1.8)    # zoom negative
mask = (a_full > -1.89) & (a_full < -1.87)  # zoom zoom negative

# Cropped a_values
a_values = a_full[mask]           # shape (N,)

# -------------------- Load full results --------------------
results_up_full   = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20.npy')
results_down_full = np.load(f'/scratch/almo2783/scratch/sweeps-test/python/deflections-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20.npy')

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
# ax.axvline(x=3.583749685, color='black', linestyle='--', linewidth=2,
#            label='a = 3.583749685 (analytical)')
ax.axvline(x=-1.876072391, color='black', linestyle='--', linewidth=2,
           label='a = -1.876072391 (analytical)')

# Formatting
ax.set_xlabel('a', fontsize=40, fontweight='bold')
ax.set_ylabel('Deflection (m)', fontsize=40, fontweight='bold')
ax.tick_params(axis='both', labelsize=25)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=20, loc='center left')
ax.margins(0)
ax.set_autoscale_on(False)

# Save figure
plt.tight_layout()
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-positive-zoom.png'
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-positive-zoom-zoom.png'
# outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-negative-zoom.png'
outname = f'extrema-odeint-sweeps-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-more-20-negative-zoom-zoom.png'
plt.savefig(outname, bbox_inches='tight')
plt.close()
