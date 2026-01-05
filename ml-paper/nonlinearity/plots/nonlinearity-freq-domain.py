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
    rho, _ = pearsonr(x, y)
    mi = mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False)[0]
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


# =====================================================
# --- Frequency-Resolved Bicoherence ---
# =====================================================

def bicoherence_spectrum(x, y, fs=1e6, nfft=4096, overlap=0.5, fmax=24000):
    """
    Compute the magnitude-squared bicoherence between x and y across frequencies.
    Returns (freqs, bicoh2) where bicoh2[f1,f2] = coupling strength.
    """
    Xseg, _ = segment_fft(x, nfft, fs, overlap=overlap)
    Yseg, _ = segment_fft(y, nfft, fs, overlap=overlap)
    nseg = min(Xseg.shape[0], Yseg.shape[0])
    nfft = Xseg.shape[1]
    max_bin = int((fmax / fs) * nfft)
    freqs = np.linspace(0, fmax, max_bin)

    num = np.zeros((max_bin, max_bin), dtype=complex)
    den1 = np.zeros((max_bin, max_bin))
    den2 = np.zeros((max_bin, max_bin))

    for s in range(nseg):
        X, Y = Xseg[s], Yseg[s]
        for f1 in range(max_bin):
            for f2 in range(max_bin):
                f3 = f1 + f2
                if f3 < max_bin:
                    num[f1, f2] += X[f1] * X[f2] * np.conj(Y[f3])
                    den1[f1, f2] += abs(X[f1] * X[f2]) ** 2
                    den2[f1, f2] += abs(Y[f3]) ** 2

    bicoh2 = abs(num) ** 2 / (den1 * den2 + 1e-20)
    return freqs, bicoh2


# =====================================================
# --- Frequency-Resolved Tricoherence ---
# =====================================================

def tricoherence_spectrum(x, y, fs=1e6, nfft=2048, overlap=0.5, fmax=24000, max_display_bins=200):
    """
    Compute magnitude-squared tricoherence across frequencies.
    Returns (freqs, tricoh2_slice) where we take a diagonal frequency slice for visualization.
    The full 3D array is huge, so we restrict or slice it for practical computation.
    """
    Xseg, _ = segment_fft(x, nfft, fs, overlap=overlap)
    Yseg, _ = segment_fft(y, nfft, fs, overlap=overlap)
    nseg = min(Xseg.shape[0], Yseg.shape[0])
    nfft = Xseg.shape[1]
    max_bin = int((fmax / fs) * nfft)
    freqs = np.linspace(0, fmax, max_bin)

    # optional reduced grid for speed/visualization
    max_bin = min(max_bin, max_display_bins)
    num = np.zeros((max_bin, max_bin), dtype=complex)
    den1 = np.zeros((max_bin, max_bin))
    den2 = np.zeros((max_bin, max_bin))

    for s in range(nseg):
        X, Y = Xseg[s], Yseg[s]
        for f1 in range(max_bin):
            for f2 in range(max_bin):
                f3 = f1
                f4 = f1 + f2 + f3
                if f4 < len(Y):
                    num[f1, f2] += X[f1]*X[f2]*X[f3]*np.conj(Y[f4])
                    den1[f1, f2] += abs(X[f1]*X[f2]*X[f3])**2
                    den2[f1, f2] += abs(Y[f4])**2

    tricoh2 = abs(num)**2 / (den1 * den2 + 1e-20)
    return freqs[:max_bin], tricoh2


# =====================================================
# --- Wrapper ---
# =====================================================

def measure_signal_nonlinearity_spectra(x, y, fs=1e6):
    """Compute correlation, MI, NLR, and frequency-resolved bi/tri coherence."""
    rho, mi, nlr = correlation_and_mi(x, y)
    freqs_bi, bicoh_map = bicoherence_spectrum(x, y, fs)
    freqs_tri, tricoh_map = tricoherence_spectrum(x, y, fs)
    return rho, mi, nlr, (freqs_bi, bicoh_map), (freqs_tri, tricoh_map)


# =====================================================
# --- Parallel Execution ---
# =====================================================

def compute_full_nonlinearity_spectra(sim_dir, save_dir, a_values, fs=1e6, n_jobs=64):
    """
    Compute and save full frequency-resolved measures for all 'a' values.
    """
    x = np.load(os.path.join(sim_dir, "original_signal.npz"))["signal"]

    def process_single_a(a):
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            return np.nan, np.nan, np.nan, None, None
        y = np.load(sim_path)["u_ac"]
        return measure_signal_nonlinearity_spectra(x, y, fs)

    results = Parallel(n_jobs=n_jobs, verbose=10, backend='threading')(
        delayed(process_single_a)(a) for a in a_values
    )

    # Extract scalar measures
    rho_all, mi_all, nlr_all = [], [], []
    bicoh_freqs, tricoh_freqs = None, None
    bicoh_all, tricoh_all = [], []

    for res in results:
        rho, mi, nlr, bicoh, tricoh = res
        rho_all.append(rho)
        mi_all.append(mi)
        nlr_all.append(nlr)
        if bicoh is not None:
            bicoh_freqs, bicoh_map = bicoh
            bicoh_all.append(bicoh_map)
        if tricoh is not None:
            tricoh_freqs, tricoh_map = tricoh
            tricoh_all.append(tricoh_map)

    # Convert to arrays
    rho_all, mi_all, nlr_all = map(np.array, [rho_all, mi_all, nlr_all])
    bicoh_all, tricoh_all = np.array(bicoh_all), np.array(tricoh_all)

    # Save scalar metrics
    np.save(os.path.join(save_dir, "linear-pearson.npy"), rho_all)
    np.save(os.path.join(save_dir, "nonlinear-MI.npy"), mi_all)
    np.save(os.path.join(save_dir, "nonlinear-ratio.npy"), nlr_all)

    # Save frequency-resolved spectra (per a)
    np.savez(os.path.join(save_dir, "bicoherence_spectra.npz"),
             freqs=bicoh_freqs, bicoh=bicoh_all, a_values=a_values)
    np.savez(os.path.join(save_dir, "tricoherence_spectra.npz"),
             freqs=tricoh_freqs, tricoh=tricoh_all, a_values=a_values)

    print("✅ Frequency-resolved results were saved.")
    return rho_all, mi_all, nlr_all, bicoh_all, tricoh_all


# =====================================================
# --- Run ---
# =====================================================

sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results2"
a_values = np.linspace(-1, 1, 101)

rho, mi, nlr, bicoh, tricoh = compute_full_nonlinearity_spectra(
    sim_dir, save_dir, a_values, fs=1e6, n_jobs=64
)
