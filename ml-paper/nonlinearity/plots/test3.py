import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# a_value = 0.44
a_values = np.array([-1.  , -0.98, -0.96, -0.94, -0.92, -0.9 , -0.88, -0.86, -0.84,
                     -0.82, -0.8 , -0.78, -0.76, -0.74, -0.72, -0.7 , -0.68, -0.66,
                     -0.64, -0.62, -0.6 , -0.58, -0.56, -0.54, -0.52, -0.5 , -0.48,
                     -0.46, -0.44, -0.42, -0.4 , -0.38, -0.36, -0.34, -0.32, -0.3 ,
                     -0.28, -0.26, -0.24, -0.22, -0.2 , -0.18, -0.16, -0.14, -0.12,
                     -0.1 , -0.08, -0.06, -0.04, -0.02,  0.  ,  0.02,  0.04,  0.06,
                      0.08,  0.1 ,  0.12,  0.14,  0.16,  0.18,  0.2 ,  0.22,  0.24,
                      0.26,  0.28,  0.3 ,  0.32,  0.34,  0.36,  0.38,  0.4 ,  0.42,
                      0.44,  0.46,  0.48,  0.5 ,  0.52,  0.54,  0.56,  0.58,  0.6 ,
                      0.62,  0.64,  0.66,  0.68,  0.7 ,  0.72,  0.74,  0.76,  0.78,
                      0.8 ,  0.82,  0.84,  0.86,  0.88,  0.9 ,  0.92,  0.94,  0.96,
                      0.98,  1.  ])
u_dc_value = 0.4
lambda_value = 1e4
mu = 1.0

# Define directories
# sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"  # Or use results_dir if preferred
results_dir = f"/scratch/almo2783/scratch/dim-less/8sensors/2600/results"
os.makedirs(save_dir, exist_ok=True)

# # Pre-load signals and compute FFTs once (efficiency gain)
# original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
# original_signal = original_data["signal"]
# fft_original_full = np.abs(np.fft.rfft(original_signal))

# sim_path = os.path.join(sim_dir, f"u_ac_a_{a_value:.2f}.npz")
# if not os.path.exists(sim_path):
#     raise FileNotFoundError(f"Missing file for a = {a_value:.2f}: {sim_path}")

# sim_data = np.load(sim_path)
# u_ac = sim_data["u_ac"]
# fft_u_ac_full = np.abs(np.fft.rfft(u_ac))

# # Run in parallel: Now passes precomputed FFTs
# correlations = Parallel(n_jobs=-1, backend='threading', verbose=10)(  # adjust n_jobs depending on your cluster
#     delayed(process_top_k)(
#         top_k, fft_original_full, fft_u_ac_full, weights
#     )
#     for idx, top_k in enumerate(top_k_values)
# )

# correlations = np.array(correlations)

# Load saved validation accuracies from results files
train_accuracies = []
val_accuracies = []
for a_value in a_values:
    results_file = f"{results_dir}/results_val-a-{a_value}-u_dc-{u_dc_value}-mu-{mu}-lam-{lambda_value:.0e}.txt"
    # /scratch/almo2783/scratch/dim-less/8sensors/2600/results/results_val-a--0.1-u_dc-0.4-mu-1.0-lam-1e+04.txt
    if os.path.exists(results_file):
        loaded_results = np.loadtxt(results_file)
        train_acc = loaded_results[3]
        val_acc = loaded_results[4]  # Validation accuracy (index 4 in [a, u_dc, lam, train_acc, val_acc, prec, rec, f1])
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    else:
        print(f"⚠️ Missing results file for a = {a_value}")
        train_accuracies.append(np.nan)
        val_accuracies.append(np.nan)

train_accuracies = np.array(train_accuracies)
val_accuracies = np.array(val_accuracies)

# Save correlations
# np.save(os.path.join(save_dir, "correlations_fft_topk.npy"), correlations)
np.save(os.path.join(save_dir, "train_accuracies.npy"), train_accuracies)
np.save(os.path.join(save_dir, "val_accuracies.npy"), val_accuracies)

# Plot
plt.figure(figsize=(10, 6))
# plt.plot(top_k_values, correlations, marker="o", color='blue', linewidth=2, label="Correlation (FFT)")
plt.plot(a_values, train_accuracies, marker="s", color='blue', linewidth=2, label="train")
plt.plot(a_values, val_accuracies, marker="s", color='darkorange', linewidth=2, label="Validation")
# plt.axhline(y=0.463571, color='blue', linestyle='--', alpha=0.7)
plt.axhline(y=0.7850, color='red', linestyle='--', alpha=0.7)
# plt.xscale('log')

plt.title(f"Accuracy vs a values (u_dc = {u_dc_value})", fontsize=20, fontweight="bold")
plt.xlabel("a values", fontsize=20, fontweight="bold")
plt.ylabel("Accuracy", fontsize=20, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontweight="bold", fontsize=18)
plt.yticks(fontweight="bold", fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

save_path = os.path.join(save_dir, f"accuracy_vs_feedback.png")
plt.savefig(save_path, dpi=300)
plt.close()