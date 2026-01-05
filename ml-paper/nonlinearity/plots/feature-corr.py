# import sys
# import time
# import os
# import numpy as np
# from joblib import Parallel, delayed
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler


# def top_features(W, top_k=10):
#     # Exclude bias (last row of W)
#     W_no_bias = W[:-1, :]

#     # Dict to store max importance per feature across all classes
#     feature_importance = {}

#     # Loop over each class (columns of W)
#     for c in range(W_no_bias.shape[1]):
#         importance = np.abs(W_no_bias[:, c])
#         top_idx = np.argsort(importance)[::-1][:top_k]

#         # Save the *maximum importance* seen across classes
#         for idx in top_idx:
#             if idx not in feature_importance:
#                 feature_importance[idx] = importance[idx]
#             else:
#                 feature_importance[idx] = max(feature_importance[idx], importance[idx])

#     # Sort collected unique indices by importance (descending)
#     sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

#     # Pick exactly top_k
#     top_idx_final = np.array([idx for idx, _ in sorted_features[:top_k]])
    
#     return top_idx_final


# def process_top_k(top_k, sim_dir, a, weights):

#     extracted_features_idx = top_features(weights, top_k=top_k)
#     print(f'top_k={top_k}, Length of unique indices = {len(extracted_features_idx)}')
    
#     # Load the reference/original signal
#     original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
#     original_signal = original_data["signal"]
#     fft_original = np.abs(np.fft.rfft(original_signal))
#     fft_original = fft_original[extracted_features_idx]

#     # correlations = []

#     # for a in a_values:
#     sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
#     if not os.path.exists(sim_path):
#         print(f"⚠️ Missing file for a = {a:.2f}")
#         correlations.append(np.nan)
#         continue

#     sim_data = np.load(sim_path)
#     u_ac = sim_data["u_ac"]
#     fft_u_ac = np.abs(np.fft.rfft(u_ac))

#     # Match lengths
#     N = min(len(fft_original), len(fft_u_ac))
#     if N < 2:
#         correlations.append(np.nan)
#         continue

#     corr = np.corrcoef(fft_original[:N], fft_u_ac[:N])[0, 1]
#     # correlations.append(corr)

#     return corr



# if __name__ == "__main__":
#     a_value = 0.44
#     u_dc_value = 0.4
#     lambda_value = 1e4
#     top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
#     mu = 1.0

#     # Results dir
#     results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/abs-weights/2600/results"
#     os.makedirs(results_dir, exist_ok=True)

#     # Load the weights
#     weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-{a_value}-lambda-{lambda_value}.npz")['arr_0']

#     # Run in parallel
#     correlations = Parallel(n_jobs=-1)(  # adjust n_jobs depending on your cluster
#         delayed(process_top_k)(
#             top_k, weights
#         )
#         for idx, top_k in enumerate(top_k_values)
#     )

#     # Save correlations
#     np.save(os.path.join(save_dir, "correlations_fft_topk.npy"), correlations)

#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(top_k_values, correlations, marker="o", linewidth=2, label="Correlation (FFT)")

#     plt.title(f"Correlation vs topk (a = {a_value}, u_dc = {u_dc_value})", fontsize=20, fontweight="bold")
#     plt.xlabel("Top K values", fontsize=20, fontweight="bold")
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.xticks(fontweight="bold", fontsize=18)
#     plt.yticks(fontweight="bold", fontsize=18)
#     plt.legend(fontsize=14)
#     plt.tight_layout()

#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"correlation_vs_topk.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()



import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


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


def process_top_k(top_k, fft_original_full, fft_u_ac_full, weights):
    extracted_features_idx = top_features(weights, top_k=top_k)
    print(f'top_k={top_k}, Length of unique indices = {len(extracted_features_idx)}')
    
    # Extract magnitudes for selected features (indices assumed to correspond to freq bins)
    fft_original_selected = fft_original_full[extracted_features_idx]
    fft_u_ac_selected = fft_u_ac_full[extracted_features_idx]

    corr = np.corrcoef(fft_original_selected, fft_u_ac_selected)[0, 1]
    return corr


if __name__ == "__main__":
    a_value = 0.44
    u_dc_value = 0.4
    lambda_value = 1e4
    top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
    mu = 1.0

    # Define directories
    sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"  # Or use results_dir if preferred
    results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/abs-weights/2600/results"
    os.makedirs(save_dir, exist_ok=True)

    # Pre-load signals and compute FFTs once (efficiency gain)
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    original_signal = original_data["signal"]
    fft_original_full = np.abs(np.fft.rfft(original_signal))

    sim_path = os.path.join(sim_dir, f"u_ac_a_{a_value:.2f}.npz")
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing file for a = {a_value:.2f}: {sim_path}")

    sim_data = np.load(sim_path)
    u_ac = sim_data["u_ac"]
    fft_u_ac_full = np.abs(np.fft.rfft(u_ac))

    # Load the weights
    weights = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/weights/weights-a-{a_value}-lambda-{lambda_value}.npz")['arr_0']

    # Run in parallel: Now passes precomputed FFTs
    correlations = Parallel(n_jobs=-1, backend='threading', verbose=10)(  # adjust n_jobs depending on your cluster
        delayed(process_top_k)(
            top_k, fft_original_full, fft_u_ac_full, weights
        )
        for idx, top_k in enumerate(top_k_values)
    )

    correlations = np.array(correlations)

    # Load saved validation accuracies from results files
    val_accuracies = []
    for top_k in top_k_values:
        results_file = f"{results_dir}/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt"
        if os.path.exists(results_file):
            loaded_results = np.loadtxt(results_file)
            val_acc = loaded_results[4]  # Validation accuracy (index 4 in [a, u_dc, lam, train_acc, val_acc, prec, rec, f1])
            val_accuracies.append(val_acc)
        else:
            print(f"⚠️ Missing results file for top_k = {top_k}")
            val_accuracies.append(np.nan)

    val_accuracies = np.array(val_accuracies)

    # Save correlations
    np.save(os.path.join(save_dir, "correlations_fft_topk.npy"), correlations)
    np.save(os.path.join(save_dir, "val_accuracies_topk.npy"), val_accuracies)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_values, correlations, marker="o", color='blue', linewidth=2, label="Correlation (FFT)")
    plt.plot(top_k_values, val_accuracies, marker="s", color='orange', linewidth=2, label="Validation")
    plt.axhline(y=0.463571, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(y=0.7850, color='orange', linestyle='--', alpha=0.7)
    plt.xscale('log')

    plt.title(f"Correlation vs topk (a = {a_value}, u_dc = {u_dc_value})", fontsize=20, fontweight="bold")
    plt.xlabel("Top K values", fontsize=20, fontweight="bold")
    plt.ylabel("Correlation", fontsize=20, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight="bold", fontsize=18)
    plt.yticks(fontweight="bold", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"correlation_vs_topk.png")
    plt.savefig(save_path, dpi=300)
    plt.close()