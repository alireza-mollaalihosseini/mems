import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch

filepath = "/scratch/almo2783/scratch/dim-less/grid-search/deflections-up/deflections-RK4-up-t-sim-500-t-rec-3-noise-2e-04-u_max-1.0-u_dc-0.76.npy"
data = np.load(filepath)
# data = data[-10000:]

# subtract the mean
data = data - np.mean(data)

a_values = np.linspace(-3, 4, 1001)

# --- Sampling frequency ---
fs = 1.0 / 1e-6

# --- Loop over signals ---
psd_results = {}
for i, a in enumerate(a_values):
    signal = data[i]

    # Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=min(4096, len(signal)))

    # --- Mask frequencies up to 2000 Hz ---
    freq_mask = freqs <= 30000
    freqs_masked = freqs[freq_mask]
    psd_masked = psd[freq_mask]

    psd_results[a] = (freqs_masked, psd_masked)

    # print(f"Computed PSD for a={a} (signal length={len(signal)})")

# --- Example plot for a few selected values ---
plt.figure(figsize=(12, 6))
for a in [-1.88, -1.87605, 3.585, 4.0]:
    if a in psd_results:
        freqs, psd = psd_results[a]
        plt.semilogy(freqs, psd, label=f"a={a}")

plt.xlabel("Frequency [Hz]", fontsize=14, fontweight="bold")
plt.ylabel("PSD [V**2/Hz]", fontsize=14, fontweight="bold")
plt.title("Power Spectral Density for selected a_values", fontsize=16, fontweight="bold")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig('PSD-Welch-masked-30000-subtract-mean.png')
plt.close()