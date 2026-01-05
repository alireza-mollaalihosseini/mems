import numpy as np
from joblib import Parallel, delayed


def fft_rank_frequencies_single_sample(
    signal,
    n_windows=10
):
    """
    signal: (T,) full raw timeseries (~1e6 points)

    Returns:
      ranked_freq_idx : (n_freqs,)
      ranked_std      : (n_freqs,)
      ranked_mean     : (n_freqs,)
    """

    T = signal.shape[0]
    win_size = T // n_windows

    ffts = []

    for w in range(n_windows):
        s = w * win_size
        e = s + win_size

        seg = signal[s:e]

        fft_mag = np.abs(np.fft.rfft(seg))
        ffts.append(fft_mag)

    # shape = (n_windows, n_freqs)
    ffts = np.stack(ffts, axis=0)

    # compute statistics across windows
    freq_mean = np.mean(ffts, axis=0)
    freq_std  = np.std(ffts, axis=0)

    # rank frequencies by temporal variation
    rank_idx = np.argsort(freq_std)[::-1]

    # apply ranking
    ranked_freq_idx = rank_idx
    ranked_std      = freq_std[rank_idx]
    ranked_mean     = freq_mean[rank_idx]

    return ranked_freq_idx, ranked_std, ranked_mean


def fft_rank_frequencies_dataset(
    X,
    fs,
    n_windows=10,
    n_jobs=8
):
    """
    X: (n_samples, T)

    Returns:
      freq_rankings : (n_samples, n_freq)
      freq_stds     : (n_samples, n_freq)
      freq_means    : (n_samples, n_freq)
    """
    T = X.shape[1]
    win_size = T // n_windows

    # physical frequency axis (shared across samples)
    freq_axis = np.fft.rfftfreq(win_size, d=1/fs)

    print("win_size:", win_size)
    print("fs:", fs)
    print("Expected step (Hz):", fs / win_size)
    print("Actual freq_axis[1] (Hz):", freq_axis[1])
    print("Nyquist (Hz):", freq_axis[-1])

    outputs = Parallel(n_jobs=n_jobs, verbose=5, backend="threading")(
        delayed(fft_rank_frequencies_single_sample)(
            X[i],
            n_windows=n_windows
        )
        for i in range(X.shape[0])
    )

    # unpack
    freq_rankings = np.stack([o[0] for o in outputs])
    freq_stds     = np.stack([o[1] for o in outputs])
    freq_means    = np.stack([o[2] for o in outputs])

    return freq_axis, freq_rankings, freq_stds, freq_means



# controls
N_WINDOWS = 10          
N_JOBS    = 64           

train_matrix = np.load(
    "/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/"
    "state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz"
)['arr_0']

print("Ranking frequencies by window variance...")

# Correct physical fs (exact: 1e6 / omega_0)
omega_0 = 16336.281798666923
FS = 1e6 / omega_0  # ≈61.213 Hz

freq_axis, freq_idx, freq_std, freq_mean = fft_rank_frequencies_dataset(
    train_matrix,
    fs=FS,
    n_windows=N_WINDOWS,
    n_jobs=N_JOBS
)

# Print diagnostics to confirm
win_size = train_matrix.shape[1] // N_WINDOWS
print(f"win_size: {win_size}")
print(f"FS: {FS}")
print(f"Expected step (Hz): {FS / win_size}")
print(f"Actual freq_axis[1] (Hz): {freq_axis[1]}")
print(f"Nyquist (Hz): {freq_axis[-1]}")

out_file = (
    "/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/"
    "fft_frequency_ranking_train.npz"
)

np.savez(
    out_file,
    freq_axis=freq_axis,              
    ranked_freq_indices=freq_idx,
    ranked_std=freq_std,
    ranked_mean=freq_mean,
    fs=FS,
    n_windows=N_WINDOWS
)