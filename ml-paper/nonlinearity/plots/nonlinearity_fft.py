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


def measure_signal_nonlinearity(x, y):
    """Compute all metrics for a single pair of signals."""
    rho, mi, nlr = correlation_and_mi(x, y)
    return rho, mi, nlr


# =====================================================
# --- Main Function ---
# =====================================================

def compute_full_nonlinearity(sim_dir, save_dir, a_values, n_jobs=64):
    """
    Compute and save linear correlation, MI, NLR, bicoherence, and tricoherence
    for all a_values in parallel.
    """
    # Load reference signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    x = original_data["signal"]
    fft_original = np.abs(np.fft.rfft(x))

    def process_single_a(a):
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            return np.nan, np.nan, np.nan
        sim_data = np.load(sim_path)
        y = sim_data["u_ac"]
        fft_u_ac = np.abs(np.fft.rfft(y))
        return measure_signal_nonlinearity(fft_original, fft_u_ac)

    # Run parallel
    results = Parallel(n_jobs=n_jobs, verbose=10, backend='threading')(
        delayed(process_single_a)(a) for a in a_values
    )

    results = np.array(results)
    rho_all, mi_all, nlr_all = results.T

    # save
    np.save(os.path.join(save_dir, "linear-pearson-fft.npy"), rho_all)
    np.save(os.path.join(save_dir, "nonlinear-MI-fft.npy"), mi_all)
    np.save(os.path.join(save_dir, "nonlinear-ratio-fft.npy"), nlr_all)

    print("✅ results were saved")
    return rho_all, mi_all, nlr_all


sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
a_values = np.linspace(-1, 1, 101)

rho, mi, nlr = compute_full_nonlinearity(sim_dir, save_dir, a_values, n_jobs=64)
