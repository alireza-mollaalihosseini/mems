import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
plt.style.use("ggplot")

# -------------------------------------------------
# Load audio
# -------------------------------------------------
wav_path = "/home/almo2783/work/10class-dataset/train/airport-barcelona-1-39-1-a.wav"   # <-- your file
fs, signal = wavfile.read(wav_path)

# Convert to float
signal = signal.astype(np.float32)

# If stereo → mono
if signal.ndim > 1:
    print("The audio is stereo(converting to mono).")
    signal = np.mean(signal, axis=1)

# Normalize
signal /= np.max(np.abs(signal) + 1e-12)

time = np.arange(len(signal)) / fs

# -------------------------------------------------
# Plot time-domain signal
# -------------------------------------------------
plt.figure(figsize=(16, 8))
plt.plot(time, signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Audio Signal (Time Domain)")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("time-domain-signal.png", dpi=300)
plt.close()

# -------------------------------------------------
# Wavelet Packet Decomposition
# -------------------------------------------------
wavelet = "db4"
maxlevel = 5   # controls TF resolution

wp = pywt.WaveletPacket(
    data=signal,
    wavelet=wavelet,
    mode="symmetric",
    maxlevel=maxlevel
)

# -------------------------------------------------
# Extract terminal nodes (leaf nodes)
# -------------------------------------------------
nodes = wp.get_level(maxlevel, order="freq")

node_labels = [node.path for node in nodes]
coeffs = np.array([node.data for node in nodes])

# Pad coefficients to same length (for plotting)
max_len = max(len(c) for c in coeffs)
coeffs_padded = np.array([
    np.pad(c, (0, max_len - len(c))) for c in coeffs
])

# # -------------------------------------------------
# # Plot Wavelet Packet Coefficients
# # -------------------------------------------------
# for i, c in enumerate(coeffs_padded):
#     plt.figure(figsize=(16, 8))
#     plt.plot(coeffs_padded[i])
#     plt.xlabel("Time Index")
#     plt.ylabel("Wavelet Packet Subband")
#     plt.title(f"Wavelet Packet Coefficient (Terminal Node: {node_labels[i]})")
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f"wavelet-packet-coefficient-node-label-{i}.png", dpi=300)
#     plt.close()

# -------------------------------------------------
# Plot Wavelet Packet Coefficients (TF Plane)
# -------------------------------------------------
plt.figure(figsize=(16, 8))
plt.imshow(
    np.abs(coeffs_padded),
    aspect="auto",
    origin="lower",
    interpolation="nearest"
)
plt.colorbar(label="|Coefficient|")
plt.yticks(range(len(node_labels)), node_labels)
plt.xlabel("Time Index")
plt.ylabel("Wavelet Packet Subband")
plt.title("Wavelet Packet Coefficients (Terminal Nodes)")
plt.tight_layout()
# plt.show()
plt.savefig("wavelet-packet-coefficients.png", dpi=300)
plt.close()