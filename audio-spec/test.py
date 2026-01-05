import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = np.load('/scratch/almo2783/scratch/audio-spec/state-matrix-train.npz')['arr_0']
data_test  = np.load('/scratch/almo2783/scratch/audio-spec/state-matrix-test.npz')['arr_0']
data_val   = np.load('/scratch/almo2783/scratch/audio-spec/state-matrix-val.npz')['arr_0']

# Sampling info
sr = 44100
fft_len = data_train.shape[1]
freqs = np.linspace(0, sr / 2, fft_len)

# Compute mean and standard deviation of magnitudes
train_mean = np.mean(data_train, axis=0)
test_mean  = np.mean(data_test, axis=0)
val_mean   = np.mean(data_val, axis=0)

train_std = np.std(data_train, axis=0)
test_std  = np.std(data_test, axis=0)
val_std   = np.std(data_val, axis=0)

# Plot each as a separate line plot
plt.figure(figsize=(16, 8))

plt.plot(freqs, train_mean, label="Train Mean", color='tab:blue')
plt.fill_between(freqs, train_mean - train_std, train_mean + train_std, color='tab:blue', alpha=0.2)

plt.axhline(y=0.0, color='r', linestyle='--', alpha=0.7)

plt.plot(freqs, test_mean, label="Test Mean", color='tab:orange')
plt.fill_between(freqs, test_mean - test_std, test_mean + test_std, color='tab:orange', alpha=0.2)

plt.plot(freqs, val_mean, label="Validation Mean", color='tab:green')
plt.fill_between(freqs, val_mean - val_std, val_mean + val_std, color='tab:green', alpha=0.2)

plt.yscale('log') 
plt.xscale('log')

plt.title("Overall Frequency Spectrum Across Datasets", fontweight="bold", fontsize=20)
plt.xlabel("Frequency (Hz)", fontweight="bold", fontsize=20)
plt.ylabel("FFT Magnitude", fontweight="bold", fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('all-spectrum.png', dpi=300)
# plt.savefig('Train-spectrum.png', dpi=300)
plt.close()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # === Load state matrix ===
# data_train = np.load('/scratch/almo2783/scratch/audio-spec/state-matrix-train.npz')['arr_0']

# # === Sampling info ===
# sr = 44100
# fft_len = data_train.shape[1]
# freqs = np.linspace(0, sr / 2, fft_len)

# # === Compute statistics ===
# train_mean = np.mean(data_train, axis=0)
# train_std = np.std(data_train, axis=0)

# # np.save('Frequencies.npy', freqs)
# # np.save('Mean_magnitudes.npy', train_mean)

# # === Prevent issues with log scale (avoid log(0)) ===
# eps = 1e-12
# train_mean = np.clip(train_mean, eps, None)
# train_std = np.clip(train_std, eps, None)

# # === Plot ===
# plt.figure(figsize=(16, 8))

# plt.plot(freqs, train_mean, label="Train Mean Spectrum", color='tab:blue')
# plt.fill_between(freqs, train_mean - train_std, train_mean + train_std, color='tab:blue', alpha=0.2)

# # plt.axhline(y=eps, color='r', linestyle='--', alpha=0.7, label='Zero Reference')

# plt.yscale('log')  # logarithmic magnitude scale
# plt.xscale('log')  # (optional) logarithmic frequency axis for better visibility of low frequencies

# plt.title("Overall Frequency Spectrum (Train Dataset)", fontweight="bold", fontsize=20)
# plt.xlabel("Frequency (Hz)", fontweight="bold", fontsize=18)
# plt.ylabel("FFT Magnitude (log scale)", fontweight="bold", fontsize=18)
# plt.legend(fontsize=14)
# plt.grid(True, which='both', alpha=0.3)
# plt.tight_layout()

# plt.savefig('Train-spectrum.png', dpi=300)
# plt.close()
