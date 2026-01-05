import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch

filepath = "/scratch/almo2783/scratch/dim-less/grid-search/deflections-up/deflections-RK4-up-t-sim-500-t-rec-3-noise-2e-04-u_max-1.0-u_dc-0.76.npy"
data = np.load(filepath)
# data = data[-10000:]


a_values = np.linspace(-3, 4, 1001)

# --- Sampling frequency ---
fs = 1.0 / 1e-6

# --- Compute PSDs and stack ---
psd_list = []
for i, a in enumerate(a_values):
    signal = data[i]
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    psd_list.append(psd)

psd_matrix = np.array(psd_list)  # shape: (len(a_values), len(freqs))

# --- Mask frequencies up to 2000 Hz ---
freq_mask = freqs <= 10000
freqs_masked = freqs[freq_mask]
psd_matrix_masked = psd_matrix[:, freq_mask]

# --- Create meshgrid for pcolormesh ---
A, F = np.meshgrid(a_values, freqs_masked, indexing="ij")  # match PSD shape

# --- Plot heatmap ---
plt.figure(figsize=(12, 6))
pcm = plt.pcolormesh(
    F, A, 10 * np.log10(psd_matrix_masked + 1e-12),  # dB scale
    shading="auto", cmap="magma"
)

plt.colorbar(pcm, label="PSD [dB/Hz]")
plt.xlabel("Frequency [Hz]", fontsize=14, fontweight="bold")
plt.ylabel("a_value", fontsize=14, fontweight="bold")
plt.title("PSD Heatmap across a_values", fontsize=16, fontweight="bold")
plt.savefig('PSD-heatmap-pcolormesh-masked-10000.png')
plt.close()