# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define labels and values
# lambda_values = np.array([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
#                      1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
# train_accuracies = []
# test_accuracies = []

# for lam in lambda_values:
#     for fold in range(10):
#         # Load test accuracy
#         # result = np.loadtxt(f"/scratch/almo2783/scratch/audio-reg/results/results-lambda-{lam:.1e}.txt")
#         result = np.loadtxt(f"/scratch/almo2783/scratch/audio-reg/results/kfold/fold_results-lambda-{lam:.1e}-fold-{fold+1}.txt")
#         train_accuracy = result[1]
#         test_accuracy = result[2]
#         train_accuracies.append(train_accuracy * 100)
#         test_accuracies.append(test_accuracy * 100)
        
# # Highest test-accuracy value
# high_test = max(test_accuracies)
# high_lambda = lambda_values[np.argmax(test_accuracies)]

# # Plotting
# plt.figure(figsize=(16, 8))
# plt.plot(lambda_values, test_accuracies, label='Test Acc.', marker='o', color='black', linewidth=3)
# plt.plot(high_lambda, high_test, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black')

# # Customize plot
# plt.xscale('log')
# plt.xticks(fontsize=40)
# plt.yticks(fontsize=40)
# plt.legend()
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout()
# plt.savefig(f'lambda-optimization-10folds.png', bbox_inches='tight', dpi=300)
# plt.close()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
# lambda_values = np.array([
#     1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
#     1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
#     1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
#     1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18
# ])
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6])

n_folds = 10
# results_dir = "/scratch/almo2783/scratch/audio-reg/results/5cities"
results_dir = "/scratch/almo2783/scratch/audio-reg/results-log"

# -----------------------------
# Load fold data per lambda
# -----------------------------
train_means = []
train_stds  = []
test_means  = []
test_stds   = []

for lam in lambda_values:

    train_vals = []
    test_vals  = []

    for fold in range(1, n_folds + 1):

        fpath = f"{results_dir}/fold_results-lambda-{lam:.1e}-fold-{fold}.txt"
        result = np.loadtxt(fpath)

        train_vals.append(result[1] * 100.0)
        test_vals.append(result[2] * 100.0)

    # Stats over folds
    train_means.append(np.mean(train_vals))
    train_stds.append(np.std(train_vals))

    test_means.append(np.mean(test_vals))
    test_stds.append(np.std(test_vals))

train_means = np.array(train_means)
train_stds  = np.array(train_stds)
test_means  = np.array(test_means)
test_stds   = np.array(test_stds)

# -----------------------------
# Best lambda from test mean
# -----------------------------
best_idx = np.argmax(test_means)
best_lambda = lambda_values[best_idx]
best_test   = test_means[best_idx]
best_train  = train_means[best_idx]
best_test_std = test_stds[best_idx]
best_train_std = train_stds[best_idx]

print("The best parameters and values:\n")
print(f"The best lambda value: {best_lambda}")
print(f"The best test value: {best_test}")
print(f"The best train value: {best_train}")
print(f"The best test std value: {best_test_std}")
print(f"The best train std value: {best_train_std}")

# -----------------------------
# Plot with error bars
# -----------------------------
plt.figure(figsize=(16, 8))

plt.errorbar(
    lambda_values, test_means,
    yerr=test_stds,
    marker='o',
    linewidth=3,
    capsize=6,
    label="Test Accuracy"
)

plt.errorbar(
    lambda_values, train_means,
    yerr=train_stds,
    marker='s',
    linewidth=3,
    capsize=6,
    label="Train Accuracy"
)

# highlight best lambda
plt.plot(
    best_lambda,
    best_test,
    'o',
    markersize=12,
    markeredgewidth=2,
    markerfacecolor='none'
)

# Formatting
plt.xscale('log')
plt.xlabel("Ridge $\\lambda$", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()

plt.savefig("lambda-optimization-10folds-log.png", dpi=300)
plt.close()

print(f"\nBest λ = {best_lambda:.1e} → Test Accuracy = {best_test:.2f}%")