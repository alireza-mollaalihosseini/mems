import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# -----------------------------
# Configuration
# -----------------------------
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6])

n_folds = 10
results_dir = "/scratch/almo2783/scratch/ml-paper/spikes/audio/results"

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

plt.savefig("lambda-optimization-10folds.png", dpi=300)
plt.close()

print(f"\nBest λ = {best_lambda:.1e} → Test Accuracy = {best_test:.2f}%")