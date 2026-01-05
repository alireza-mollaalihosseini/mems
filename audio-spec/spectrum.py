import numpy as np
import soundfile as sf
from joblib import Parallel, delayed

# -------------------------------
# Parameters
# -------------------------------
fft_len = 22000
n_jobs = -1  # use all available CPU cores

# -------------------------------
# Helper function
# -------------------------------
def compute_fft_from_file(fname, fft_len):
    data, samplerate = sf.read(fname)
    # Optional check (since you said all are 48000 Hz)
    if samplerate != 44100:
        raise ValueError(f"Unexpected samplerate {samplerate} in file {fname}")

#     # Truncate or pad the data to ensure consistent length
#     if len(data) < fft_len:
#         data = np.pad(data, (0, fft_len - len(data)), mode='constant')
#     else:
#         data = data[:fft_len]

    # Compute magnitude of FFT
    fft_vals = np.fft.rfft(data)
    fft_magnitude = np.abs(fft_vals)

    # Pad/truncate FFT to fft_len for consistent matrix shape
    if len(fft_magnitude) < fft_len:
        fft_magnitude = np.pad(fft_magnitude, (0, fft_len - len(fft_magnitude)))
    else:
        fft_magnitude = fft_magnitude[:fft_len]

    return fft_magnitude.astype(np.float64)

# -------------------------------
# Dataset paths
# -------------------------------
train_path = '/scratch/almo2783/scratch/rayson/design1/barcelona/train-filenames-barcelona-rayson.csv'
test_path  = '/scratch/almo2783/scratch/rayson/design1/barcelona/test-filenames-barcelona-rayson.csv'
val_path   = '/scratch/almo2783/scratch/rayson/design1/barcelona/val-filenames-barcelona-rayson.csv'

# -------------------------------
# Load file lists
# -------------------------------
train_filenames = np.loadtxt(train_path, dtype=str)
test_filenames  = np.loadtxt(test_path, dtype=str)
val_filenames   = np.loadtxt(val_path, dtype=str)

# -------------------------------
# Compute FFTs in parallel
# -------------------------------
train_state_matrix = np.vstack(Parallel(n_jobs=n_jobs)(
    delayed(compute_fft_from_file)(fname, fft_len) for fname in train_filenames
))

np.savez_compressed("/scratch/almo2783/scratch/audio-spec/state-matrix-train.npz", train_state_matrix)

test_state_matrix = np.vstack(Parallel(n_jobs=n_jobs)(
    delayed(compute_fft_from_file)(fname, fft_len) for fname in test_filenames
))

np.savez_compressed("/scratch/almo2783/scratch/audio-spec/state-matrix-test.npz", test_state_matrix)

val_state_matrix = np.vstack(Parallel(n_jobs=n_jobs)(
    delayed(compute_fft_from_file)(fname, fft_len) for fname in val_filenames
))

np.savez_compressed("/scratch/almo2783/scratch/audio-spec/state-matrix-val.npz", val_state_matrix)

print("Shapes:")
print(f"Train: {train_state_matrix.shape}")
print(f"Test:  {test_state_matrix.shape}")
print(f"Val:   {val_state_matrix.shape}")
