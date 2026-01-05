import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
a_values      = np.linspace(-1, 1, 41)
n_folds       = 10
results_dir = "/scratch/almo2783/scratch/ml-paper/10fold/results/5cities"

# -----------------------------
# Outputs
# -----------------------------
best_train_a   = []
best_test_a    = []
best_lambda_a  = []
best_train_std = []   # <-- needed for correct error bars
best_test_std  = []   # <-- needed for correct error bars

# track missing files
missing_files = []

# -----------------------------
# Main sweep
# -----------------------------
for a in a_values:

    train_means = []
    train_stds  = []
    test_means  = []
    test_stds   = []

    for lam in lambda_values:

        train_vals = []
        test_vals  = []

        for fold in range(1, n_folds + 1):

            fpath = (
                f"{results_dir}/"
                f"fold_results-a-{a:.2f}-lambda-{lam:.1e}-fold-{fold}.txt"
            )

            # -----------------------------
            # Checkpoint: file exists?
            # -----------------------------
            if not os.path.exists(fpath):
                missing_files.append((a, lam, fold))
                continue

            result = np.loadtxt(fpath)

            # result expected format:
            # [ fold_id, train_acc, test_acc ]
            train_vals.append(result[1] * 100.0)
            test_vals.append(result[2] * 100.0)
            

        # Skip lambda if no folds found
        if len(train_vals) == 0:
            train_means.append(np.nan)
            train_stds.append(np.nan)
            test_means.append(np.nan)
            test_stds.append(np.nan)
            continue

        # -----------------------------
        # Stats over folds (THIS is correct)
        # -----------------------------
        train_means.append(np.mean(train_vals))
        train_stds.append(np.std(train_vals))

        test_means.append(np.mean(test_vals))
        test_stds.append(np.std(test_vals))

    # -------- robust best-lambda selection --------
    test_means = np.array(test_means)

    if np.all(np.isnan(test_means)):
        print(f"⚠️  All NaNs for a={a:.2f}, skipping")
        best_lambda_a.append(np.nan)
        best_test_a.append(np.nan)
        best_train_a.append(np.nan)
        best_test_std.append(np.nan)
        best_train_std.append(np.nan)
        continue

    # convert to arrays
    train_means = np.array(train_means)
    train_stds  = np.array(train_stds)
    test_means  = np.array(test_means)
    test_stds   = np.array(test_stds)

    # -----------------------------
    # Best lambda from test mean
    # (ignore NaNs from missing lambdas)
    # -----------------------------
    best_idx = np.nanargmax(test_means)

    best_lambda = lambda_values[best_idx]
    best_test   = test_means[best_idx]
    best_train  = train_means[best_idx]

    best_test_s  = test_stds[best_idx]
    best_train_s = train_stds[best_idx]

    # store
    best_lambda_a.append(best_lambda)
    best_test_a.append(best_test)
    best_train_a.append(best_train)

    best_test_std.append(best_test_s)
    best_train_std.append(best_train_s)


# -----------------------------
# Convert to arrays (fix typo)
# -----------------------------
best_train_a   = np.array(best_train_a)
best_test_a    = np.array(best_test_a)
best_lambda_a  = np.array(best_lambda_a)

best_train_std = np.array(best_train_std)
best_test_std  = np.array(best_test_std)

# save the values
np.save(f"best_train_over_a-5cities.npy",best_train_a)
np.save(f"best_train_std-5cities.npy",best_train_std)
np.save(f"best_test_over_a-5cities.npy",best_test_a)
np.save(f"best_test_std-5cities.npy",best_test_std)
np.save(f"best_lambda_over_a-5cities.npy",best_lambda_a)

# -----------------------------
# Plot with correct error bars
# -----------------------------
plt.figure(figsize=(16, 8))

plt.errorbar(
    a_values,
    best_test_a,
    yerr=best_test_std,
    marker='o',
    linewidth=3,
    capsize=6,
    label="Test Accuracy"
)

plt.errorbar(
    a_values,
    best_train_a,
    yerr=best_train_std,
    marker='s',
    linewidth=3,
    capsize=6,
    label="Train Accuracy"
)

# Formatting
plt.xlabel("Feedback (a)", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()

plt.savefig(f"feedback-10folds-5cities.png", dpi=300)
plt.close()


# -----------------------------
# Report missing files
# -----------------------------
if missing_files:
    print("\n==============================")
    print("⚠️  Missing result files found")
    print("==============================")
    print(f"Total missing files: {len(missing_files)}")

    # Print first N so it doesn't spam terminal
    max_print = 40
    for i, (a, lam, fold) in enumerate(missing_files[:max_print]):
        print(f"[{i+1:02d}] a={a:.3f}, lambda={lam:.1e}, fold={fold}")

    if len(missing_files) > max_print:
        print(f"... and {len(missing_files) - max_print} more")
else:
    print("\n✅ All result files loaded successfully.")
