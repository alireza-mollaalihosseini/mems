import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

filepath = "/scratch/almo2783/scratch/dim-less/grid-search/deflections-up1/deflections-RK4-up-t-sim-500-t-rec-3-noise-2e-04-u_max-1.0-u_dc-0.76.npy"
data = np.load(filepath)

data1 = data[700][-50000:]
data2 = data[470][-500:]

peaks1, _ = find_peaks(data1)
peaks2, _ = find_peaks(data2)

maxima1 = data1[peaks1]
maxima2 = data2[peaks2]

print(f'unique maximas for 700: {np.unique(maxima1.round(decimals=3))}')
print(f'unique maximas for 10: {np.unique(maxima2.round(decimals=3))}')

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].plot(np.arange(len(data1)), data1, label='700')
axes[0].scatter(peaks1, maxima1, color='red')
axes[1].plot(np.arange(len(data2)), data2, label='10')
axes[1].scatter(peaks2, maxima2, color='red')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('test-trace.png')
plt.close()