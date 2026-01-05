import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

filepath = "/scratch/almo2783/scratch/dim-less/grid-search/deflections-up1/deflections-RK4-up-t-sim-500-t-rec-3-noise-2e-04-u_max-1.0-u_dc-0.76.npy"
data = np.load(filepath)
data = data[-50000:]


# Original ranges
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

for i, a in enumerate(a_values):
    peaks, _ = find_peaks(data[i])
    maxima = data[i][peaks]
    unique_maxima = np.unique(maxima.round(decimals=4))
    if np.std(unique_maxima) < 0.01:
        unique_maxima = unique_maxima[-1]

    print(f'unique maximas for a={a}: {unique_maxima}')