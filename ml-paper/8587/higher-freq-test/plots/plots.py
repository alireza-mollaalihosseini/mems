import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


results_dir = "/scratch/almo2783/scratch/ml-paper/8587/higher-freq-test"
# f_values = np.array([2600, 8000, 8587, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
#                     17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000])
f_values = np.array([8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
                    17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000])

lambda_values = np.array([
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    1, 10, 1e2, 1e3, 1e4, 1e5, 1e6
])

# Containers
best_test = np.full(len(f_values), np.nan)
best_train = np.full(len(f_values), np.nan)
std_test = np.full(len(f_values), np.nan)
best_lambda = np.full(len(f_values), np.nan)

# # -------------------------
# # Load & analyze results
# # -------------------------
# for i, f in enumerate(f_values):

#     fname = os.path.join(
#         results_dir,
#         f"results-a-0.44-mu-1.00e+00-f-{int(f)}.txt"
#     )

#     if not os.path.exists(fname):
#         print(f"WARNING: Missing {fname}")
#         best_acc_per_mu[i] = np.nan
#         best_lambda_per_mu[i] = np.nan
#         continue

#     # columns:
#     # 0 → lambda
#     # 1 → train accuracy
#     # 2 → validation/test accuracy
#     data = np.loadtxt(fname, skiprows=1)

#     lambdas = data[:, 0]
#     train_acc = data[:, 1]
#     test_acc = data[:, 2]     # val_acc column

#     # best lambda index
#     idx_best = np.argmax(test_acc)

#     best_acc_per_mu[i] = test_acc[idx_best]
#     best_train_acc_per_mu[i] = train_acc[idx_best]
#     best_lambda_per_mu[i] = lambdas[idx_best]


# # -------------------------
# # Global maximum (over μ)
# # -------------------------
# global_best_idx = np.nanargmax(best_acc_per_mu)

# global_best_mu     = f_values[global_best_idx]
# global_best_acc    = best_acc_per_mu[global_best_idx]
# global_best_lambda = best_lambda_per_mu[global_best_idx]

# # -------------------------
# # Plot
# # -------------------------
# plt.style.use('ggplot')
# plt.figure(figsize=(16,8))
# plt.semilogx(f_values, best_acc_per_mu * 100, marker='o', label="Validation Acc.")
# plt.semilogx(f_values, best_train_acc_per_mu * 100, marker='o', label="Train Acc.")

# # Highlight best point
# plt.scatter(
#     global_best_mu,
#     global_best_acc * 100,
#     s=120,
#     marker='*',
#     zorder=5,
#     label=(
#         f"Peak val acc = {global_best_acc * 100:.4f}\n"
#         f"at f = {int(f)}"
#     )
# )

# # Vertical line at best μ
# plt.axvline(
#     global_best_mu,
#     linestyle=':',
#     linewidth=1.5
# )

# plt.xlabel('Resonance frequency of the sensor', fontweight='bold', fontsize=20)
# plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=20)

# plt.xticks(fontweight='bold', fontsize=16)
# plt.yticks(fontweight='bold', fontsize=16)

# plt.grid(True, which="both", linestyle='--', alpha=0.5)

# plt.title(
#     "Best Validation Accuracy vs f",
#     fontweight='bold',
#     fontsize=18
# )
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.legend()
# plt.tight_layout()
# plt.savefig("Val-opt-for-best-lambda.png", dpi=300)
# plt.close()


# ----------------------------------
# Load & analyze
# ----------------------------------
for i, f in enumerate(f_values):

    fname = os.path.join(
        results_dir,
        f"results-a-0.44-mu-1.00e+00-f-{int(f)}-Q_0-500.txt"
    )

    if not os.path.exists(fname):
        print(f"WARNING: Missing {fname}")
        continue

    data = np.loadtxt(fname, skiprows=1)

    lambdas   = data[:, 0]
    train_acc = data[:, 1] * 100
    test_acc  = data[:, 2] * 100

    idx_best = np.argmax(test_acc)

    best_test[i]   = test_acc[idx_best]
    best_train[i]  = train_acc[idx_best]
    best_lambda[i] = lambdas[idx_best]

    # std across λ (for uncertainty band)
    std_test[i] = np.std(test_acc)

# ----------------------------------
# Global best
# ----------------------------------
valid_mask = ~np.isnan(best_test)
global_idx = np.argmax(best_test[valid_mask])

f_best      = f_values[valid_mask][global_idx]
acc_best    = best_test[valid_mask][global_idx]
lambda_best = best_lambda[valid_mask][global_idx]

# ----------------------------------
# Plot (fill_between style)
# ----------------------------------
plt.figure(figsize=(16, 8))

# Validation accuracy
plt.plot(
    f_values,
    best_test,
    marker='o',
    linewidth=3,
    label="Validation Accuracy"
)

# plt.fill_between(
#     f_values,
#     best_test - std_test,
#     np.clip(best_test + std_test, 0, 100),
#     alpha=0.35
# )

# Training accuracy
plt.plot(
    f_values,
    best_train,
    marker='s',
    linewidth=3,
    linestyle='--',
    label="Training Accuracy"
)

# Highlight global best
plt.scatter(
    f_best,
    acc_best,
    s=200,
    marker='*',
    zorder=5,
    label=(
        f"Best val = {acc_best:.2f}%\n"
        f"f = {int(f_best)} Hz\n"
        f"λ = {lambda_best:.1e}"
    )
)

plt.axvline(f_best, linestyle=':', linewidth=2)


# Baseline (without reservoir)
baseline_mean = 83.41887000000001
baseline_std  = 1.2031135898575849
baseline_vals = np.full_like(f_values, baseline_mean)
plt.plot(f_values, baseline_vals, marker='o', label="Val. Acc. without Reservoir")
plt.fill_between(f_values, baseline_vals - baseline_std, baseline_vals + baseline_std, alpha=0.2)


# Labels
plt.xlabel("Resonance frequency of the sensor (Hz)", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)

plt.xticks(fontweight='bold', fontsize=16)
plt.yticks(fontweight='bold', fontsize=16)

plt.title(
    "Best Validation Accuracy vs Resonance Frequency\n"
    r"($a=0.44,\ \mu=1.0$)",
    fontweight='bold',
    fontsize=18
)

plt.legend(fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()

plt.savefig("Val-opt-vs-frequency-fillbetween-Q_0-500.png", dpi=300)
plt.close()