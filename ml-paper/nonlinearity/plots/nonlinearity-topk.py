import os
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import get_window
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed

# =====================================================
# --- Helper Functions ---
# =====================================================

def top_features(W, top_k=10):
    # Exclude bias (last row of W)
    W_no_bias = W[:-1, :]

    # Dict to store max importance per feature across all classes
    feature_importance = {}

    # Loop over each class (columns of W)
    for c in range(W_no_bias.shape[1]):
        importance = np.abs(W_no_bias[:, c])
        top_idx = np.argsort(importance)[::-1][:top_k]

        # Save the *maximum importance* seen across classes
        for idx in top_idx:
            if idx not in feature_importance:
                feature_importance[idx] = importance[idx]
            else:
                feature_importance[idx] = max(feature_importance[idx], importance[idx])

    # Sort collected unique indices by importance (descending)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Pick exactly top_k
    top_idx_final = np.array([idx for idx, _ in sorted_features[:top_k]])
    
    return top_idx_final


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

def compute_full_nonlinearity(sim_dir, save_dir, top_k_values, weights, n_jobs=64):
    """
    Compute and save linear correlation, MI, NLR
    for all a_values in parallel.
    """
    a_value = 0.44

    # Load reference signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    x = original_data["signal"]
    fft_original_full = np.abs(np.fft.rfft(x))

    def process_single_topk(topk, a):
        extracted_features_idx = top_features(weights, top_k=topk)
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        sim_data = np.load(sim_path)
        y = sim_data["u_ac"]
        fft_u_ac_full = np.abs(np.fft.rfft(y))
        return measure_signal_nonlinearity(fft_original_full[extracted_features_idx], fft_u_ac_full[extracted_features_idx])

    # Run parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_topk)(topk, a_value) for topk in top_k_values
    )

    results = np.array(results)
    rho_all, mi_all, nlr_all = results.T

    # save
    np.save(os.path.join(save_dir, "linear-pearson-topk.npy"), rho_all)
    np.save(os.path.join(save_dir, "nonlinear-MI-topk.npy"), mi_all)
    np.save(os.path.join(save_dir, "nonlinear-ratio-topk.npy"), nlr_all)

    print("✅ Nonlinearity results saved to 'nonlinearity_results.npz'")
    return rho_all, mi_all, nlr_all


sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]

# Load the weights
weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-0.44-lambda-10000.0.npz")['arr_0']

rho, mi, nlr = compute_full_nonlinearity(sim_dir, save_dir, top_k_values, weights, n_jobs=64)