import os
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import get_window
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed

# =====================================================
# --- Helper Functions ---
# =====================================================

def correlation_and_mi(x, y):
    """Compute Pearson correlation, mutual information, and nonlinearity ratio."""
    N = min(len(x), len(y))
    x, y = x[:N], y[:N]

    # Linear correlation
    rho, _ = pearsonr(x, y)

    # Mutual information
    mi = mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False)[0]

    # Nonlinearity ratio (normalized by |rho|)
    nlr = mi / (abs(rho) + 1e-12)

    return rho, mi, nlr


def segment_fft(signal, nfft, fs, window='hann', overlap=0.5):
    """Divide signal into overlapping segments and return FFT matrix."""
    step = int(nfft * (1 - overlap))
    win = get_window(window, nfft)
    norm = np.sum(win**2)
    segments = [
        np.fft.fft(signal[i:i+nfft] * win, nfft)
        for i in range(0, len(signal) - nfft, step)
    ]
    return np.array(segments), norm


def bicoherence(x, y, fs=1e6, nfft=4096, overlap=0.5, fmax=24000):
    """
    Compute the magnitude-squared bicoherence between x and y.
    Returns average value up to fmax.
    """
    Xseg, _ = segment_fft(x, nfft, fs, overlap=overlap)
    Yseg, _ = segment_fft(y, nfft, fs, overlap=overlap)
    nseg = min(Xseg.shape[0], Yseg.shape[0])
    nfft = Xseg.shape[1]
    max_bin = int((fmax / fs) * nfft)

    num = np.zeros((max_bin, max_bin), dtype=complex)
    den1 = np.zeros((max_bin, max_bin))
    den2 = np.zeros((max_bin, max_bin))

    for s in range(nseg):
        X, Y = Xseg[s], Yseg[s]
        for f1 in range(max_bin):
            for f2 in range(max_bin):
                f3 = (f1 + f2) % nfft
                num[f1, f2] += X[f1] * X[f2] * np.conj(Y[f3])
                den1[f1, f2] += abs(X[f1] * X[f2]) ** 2
                den2[f1, f2] += abs(Y[f3]) ** 2

    bicoh2 = abs(num) ** 2 / ((den1 * den2) + 1e-20)
    return np.nanmean(bicoh2)


def tricoherence(x, y, fs=1e6, nfft=2048, overlap=0.5, fmax=24000):
    """
    Compute the magnitude-squared tricoherence between x and y.
    Returns average value up to fmax.
    """
    Xseg, _ = segment_fft(x, nfft, fs, overlap=overlap)
    Yseg, _ = segment_fft(y, nfft, fs, overlap=overlap)
    nseg = min(Xseg.shape[0], Yseg.shape[0])
    nfft = Xseg.shape[1]
    max_bin = int((fmax / fs) * nfft)

    num = np.zeros((max_bin, max_bin, max_bin), dtype=complex)
    den1 = np.zeros((max_bin, max_bin, max_bin))
    den2 = np.zeros((max_bin, max_bin, max_bin))

    for s in range(nseg):
        X, Y = Xseg[s], Yseg[s]
        for f1 in range(max_bin):
            for f2 in range(max_bin):
                for f3 in range(max_bin):
                    f4 = (f1 + f2 + f3) % nfft
                    num[f1, f2, f3] += X[f1]*X[f2]*X[f3]*np.conj(Y[f4])
                    den1[f1, f2, f3] += abs(X[f1]*X[f2]*X[f3])**2
                    den2[f1, f2, f3] += abs(Y[f4])**2

    tricoh2 = abs(num)**2 / ((den1 * den2) + 1e-20)
    return np.nanmean(tricoh2)


def measure_signal_nonlinearity(x, y, fs=1e6):
    """Compute correlation, MI, NLR, true bicoherence and tricoherence."""
    rho, mi, nlr = correlation_and_mi(x, y)
    bicoh = bicoherence(x, y, fs)
    tricoh = tricoherence(x, y, fs)
    return rho, mi, nlr, bicoh, tricoh


# =====================================================
# --- Main Function ---
# =====================================================

def compute_full_nonlinearity(sim_dir, save_dir, a_values, fs=1e6, n_jobs=64):
    """
    Compute and save linear correlation, MI, NLR, bicoherence, and tricoherence
    for all a_values in parallel.
    """
    # Load reference signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    x = original_data["signal"]

    def process_single_a(a):
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        sim_data = np.load(sim_path)
        y = sim_data["u_ac"]
        return measure_signal_nonlinearity(x, y, fs)

    # Run parallel
    results = Parallel(n_jobs=n_jobs, verbose=10, backend='threading')(
        delayed(process_single_a)(a) for a in a_values
    )

    results = np.array(results)
    rho_all, mi_all, nlr_all, bicoh_all, tricoh_all = results.T

    # # Save
    # np.savez(
    #     os.path.join(sim_dir, "nonlinearity_results.npz"),
    #     a_values=a_values,
    #     rho=rho_all,
    #     mi=mi_all,
    #     nlr=nlr_all,
    #     bicoherence=bicoh_all,
    #     tricoherence=tricoh_all,
    #     fs=fs,
    # )

    # save
    np.save(os.path.join(save_dir, "linear-pearson.npy"), rho_all)
    np.save(os.path.join(save_dir, "nonlinear-MI.npy"), mi_all)
    np.save(os.path.join(save_dir, "nonlinear-ratio.npy"), nlr_all)
    np.save(os.path.join(save_dir, "bicoherence.npy"), bicoh_all)
    np.save(os.path.join(save_dir, "tricoherence.npy"), tricoh_all)

    print("✅ results were saved")
    return rho_all, mi_all, nlr_all, bicoh_all, tricoh_all


sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
a_values = np.linspace(-1, 1, 101)

rho, mi, nlr, bicoh, tricoh = compute_full_nonlinearity(sim_dir, save_dir, a_values, fs=1e6, n_jobs=64)
