import numpy as np
import matplotlib.pyplot as plt

# a_values = np.linspace(-2, 4, 1001)
# Original ranges
# a1 = np.arange(-2.0, -1.9, 0.1)
# a2 = np.arange(-1.9, -1.8, 0.0005)
# a3 = np.arange(-1.8, 3.5, 0.1)
# a4 = np.arange(3.5, 3.7, 0.0005)
# a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# # Additional 20 points between specified intervals
# extra1 = np.linspace(-1.8775, -1.875, 200)
# extra2 = np.linspace(3.5825, 3.585, 200)

# # Combine all and sort
# a_values = np.concatenate([a1, a2, extra1, a3, a4, extra2, a5])
# a_values = np.unique(np.round(a_values, decimals=8))

a1 = np.arange(-2.0, -1.9, 0.1)
a2 = np.arange(-1.9, -1.8, 0.0005)
a3 = np.arange(-1.8, 3.5, 0.1)
a4 = np.arange(3.5, 3.7, 0.0005)
a5 = np.arange(3.7, 4.0 + 1e-8, 0.1)

# Additional 20 points between specified intervals
extra1 = np.linspace(-1.8775, -1.875, 200)
extra2 = np.linspace(3.5825, 3.585, 200)
extra3 = np.linspace(-1.8761, -1.87605, 200)

# Combine all and sort
a_values = np.concatenate([a1, a2, extra1, extra3, a3, a4, extra2, a5])
a_values = np.unique(np.round(a_values, decimals=8))

kick = 1e-3
direction = 'up'
max_peaks = np.load(f"/scratch/almo2783/scratch/dim-less/with-lina/max_peaks-{direction}-kick-{kick}.npy")
min_peaks = np.load(f"/scratch/almo2783/scratch/dim-less/with-lina/min_peaks-{direction}-kick-{kick}.npy")

# a_full = a_values
# a_full = np.linspace(-2, 4, 1001)
# a_full = np.arange(-2.0, 4.01, 0.01)
# Choose your zoom window here:
# mask = (a_full > 3.5) & (a_full < 3.7)      # zoom positive
# mask = (a_full > 3.575) & (a_full < 3.595)     # zoom zoom positive
# mask = (a_full > -1.9) & (a_full < -1.8)    # zoom negative
# mask = (a_full > -1.89) & (a_full < -1.87)  # zoom zoom negative
# mask = (a_full > -1.8775) & (a_full < -1.875)  # zoom zoom negative 1
# mask = (a_full > -2.0) & (a_full < -1.875)
# mask = (a_full > 3.3) & (a_full < 3.8)

# # Cropped a_values
# a_values = a_full[mask]           # shape (N,)
# max_peaks = max_peaks[mask]
# min_peaks = min_peaks[mask]

# ---------------- Plot ----------------
plt.figure(figsize=(16, 8))
plt.scatter(a_values, max_peaks, label="Global max peak")
plt.scatter(a_values, min_peaks, label="Global min peak")

plt.axvline(x=3.583749685, color='black', linestyle='--', linewidth=2,
           label='a = 3.583749685 (analytical)')
plt.axvline(x=-1.876072391, color='black', linestyle='--', linewidth=2,
           label='a = -1.876072391 (analytical)')

plt.xlabel("a", fontweight="bold", fontsize=14)
plt.ylabel("Peak displacement", fontweight="bold", fontsize=14)
plt.title("Global max/min peaks vs a", fontweight="bold", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(f'bifurcation-up-kick-{kick}-more-points.png')
# plt.savefig(f'bifurcation-up-tau-kick-{kick}-negative.png')
# plt.savefig(f'bifurcation-up-tau-kick-{kick}-positive.png')
plt.close()