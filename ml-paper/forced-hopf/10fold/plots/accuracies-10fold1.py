# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Configuration
# # -----------------------------
# lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
# lam_values      = np.array([-1.0 , -0.98 ,-0.96, -0.94, -0.92 , -0.9 ,  -0.88 ,-0.86, -0.84,
#                           -0.82, -0.8  ,-0.78, -0.76, -0.74 , -0.72,  -0.7  ,-0.68 , -0.66, 
#                           -0.64, -0.62 ,-0.6 ,-0.58 , -0.56 , -0.54,  -0.52 ,-0.5 , -0.48 ,
#                           -0.46, -0.44 ,-0.42, -0.4 , -0.38 , -0.36,  -0.34 ,-0.32, -0.3 ,
#                           -0.28, -0.26 ,-0.24, -0.22, -0.2  ,-0.18 , -0.16  ,-0.14 , -0.12 ,
#                           -0.1 , -0.08 ,-0.06, -0.04, -0.02 , 0.0  , 0.02   ,0.04  , 0.06  ,0.08 ,
#                           0.1  , 0.12  ,0.14 , 0.16 , 0.18  ,0.2   ,0.22    ,0.24   ,0.26  ,0.28  ,0.3 ,
#                           0.32 , 0.34  ,0.36 , 0.38 , 0.4   ,0.42  ,0.44    ,0.46   ,0.48  ,0.5   ,0.52 ,
#                           0.54 , 0.56  ,0.58 , 0.6  , 0.62  ,0.64  ,0.66    ,0.68   ,0.7   ,0.72  ,0.74 ,
#                           0.76 , 0.78  ,0.8  , 0.82 , 0.84  ,0.86  ,0.88    ,0.9    ,0.92  ,0.94  ,0.96 ,
#                           0.98 , 1.0])
# n_folds       = 10
# alpha = -0.01
# beta  = 0.3
# mu    = 1e4 # 0.1 # 1.0
# results_dir = "/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/results"

# # -----------------------------
# # Outputs
# # -----------------------------
# best_train_lam   = []
# best_test_lam    = []
# best_lambda_lam  = []
# best_train_std = []   # <-- needed for correct error bars
# best_test_std  = []   # <-- needed for correct error bars

# # track missing files
# missing_files = []

# # -----------------------------
# # Main sweep
# # -----------------------------
# for lam in lam_values:

#     train_means = []
#     train_stds  = []
#     test_means  = []
#     test_stds   = []

#     for lambda_value in lambda_values:

#         train_vals = []
#         test_vals  = []

#         for fold in range(1, n_folds + 1):

#             fpath = (
#                 f"{results_dir}/"
#                 f"fold_results-alpha-{alpha:.2f}-beta-{beta:.2f}-lam-{lam:.2f}-mu-{mu:.0e}-lambda-ridge-{lambda_value:.1e}-fold-{fold}.txt"
#             )

#             # -----------------------------
#             # Checkpoint: file exists?
#             # -----------------------------
#             if not os.path.exists(fpath):
#                 missing_files.append((lam, lambda_value, fold))
#                 continue

#             result = np.loadtxt(fpath)

#             # result expected format:
#             # [ fold_id, train_acc, test_acc ]
#             train_vals.append(result[1] * 100.0)
#             test_vals.append(result[2] * 100.0)
            

#         # Skip lambda if no folds found
#         if len(train_vals) == 0:
#             train_means.append(np.nan)
#             train_stds.append(np.nan)
#             test_means.append(np.nan)
#             test_stds.append(np.nan)
#             continue

#         # -----------------------------
#         # Stats over folds (THIS is correct)
#         # -----------------------------
#         train_means.append(np.mean(train_vals))
#         train_stds.append(np.std(train_vals))

#         test_means.append(np.mean(test_vals))
#         test_stds.append(np.std(test_vals))

#     # -------- robust best-lambda selection --------
#     test_means = np.array(test_means)

#     if np.all(np.isnan(test_means)):
#         print(f"⚠️  All NaNs for lam={lam:.2f}, skipping")
#         best_lambda_lam.append(np.nan)
#         best_test_lam.append(np.nan)
#         best_train_lam.append(np.nan)
#         best_test_std.append(np.nan)
#         best_train_std.append(np.nan)
#         continue

#     # convert to arrays
#     train_means = np.array(train_means)
#     train_stds  = np.array(train_stds)
#     test_means  = np.array(test_means)
#     test_stds   = np.array(test_stds)

#     # -----------------------------
#     # Best lambda from test mean
#     # (ignore NaNs from missing lambdas)
#     # -----------------------------
#     best_idx = np.nanargmax(test_means)

#     best_lambda = lambda_values[best_idx]
#     best_test   = test_means[best_idx]
#     best_train  = train_means[best_idx]

#     best_test_s  = test_stds[best_idx]
#     best_train_s = train_stds[best_idx]

#     # store
#     best_lambda_lam.append(best_lambda)
#     best_test_lam.append(best_test)
#     best_train_lam.append(best_train)

#     best_test_std.append(best_test_s)
#     best_train_std.append(best_train_s)


# # -----------------------------
# # Convert to arrays (fix typo)
# # -----------------------------
# best_train_lam   = np.array(best_train_lam)
# best_test_lam    = np.array(best_test_lam)
# best_lambda_lam  = np.array(best_lambda_lam)

# best_train_std = np.array(best_train_std)
# best_test_std  = np.array(best_test_std)

# # save the values
# np.save(f"best_train_over_lam-mu-{mu:.2f}.npy",best_train_lam)
# np.save(f"best_train_std-mu-{mu:.2f}.npy",best_train_std)
# np.save(f"best_test_over_lam-mu-{mu:.2f}.npy",best_test_lam)
# np.save(f"best_test_std-mu-{mu:.2f}.npy",best_test_std)
# np.save(f"best_lambda_over_lam-mu-{mu:.2f}.npy",best_lambda_lam)

# # -----------------------------
# # Plot with correct error bars
# # -----------------------------
# plt.figure(figsize=(16, 8))

# plt.errorbar(
#     lam_values,
#     best_test_lam,
#     yerr=best_test_std,
#     marker='o',
#     linewidth=3,
#     capsize=6,
#     label="Test Accuracy"
# )

# plt.errorbar(
#     lam_values,
#     best_train_lam,
#     yerr=best_train_std,
#     marker='s',
#     linewidth=3,
#     capsize=6,
#     label="Train Accuracy"
# )

# # Formatting
# plt.xlabel("Lambda", fontweight='bold', fontsize=20)
# plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.legend(fontsize=18)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()

# plt.savefig(f"lambda-optimization-10folds-errorbars-with-best-lambdas-mu-{mu:.0e}.png", dpi=300)
# plt.close()


# # -----------------------------
# # Report missing files
# # -----------------------------
# if missing_files:
#     print("\n==============================")
#     print("⚠️  Missing result files found")
#     print("==============================")
#     print(f"Total missing files: {len(missing_files)}")

#     # Print first N so it doesn't spam terminal
#     max_print = 40
#     for i, (lam, lambda_value, fold) in enumerate(missing_files[:max_print]):
#         print(f"[{i+1:02d}] a={lam:.2f}, lambda={lambda_value:.1e}, fold={fold}")

#     if len(missing_files) > max_print:
#         print(f"... and {len(missing_files) - max_print} more")
# else:
#     print("\n✅ All result files loaded successfully.")


import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
lam_values      = np.array([-1.0 , -0.98 ,-0.96, -0.94, -0.92 , -0.9 ,  -0.88 ,-0.86, -0.84,
                          -0.82, -0.8  ,-0.78, -0.76, -0.74 , -0.72,  -0.7  ,-0.68 , -0.66, 
                          -0.64, -0.62 ,-0.6 ,-0.58 , -0.56 , -0.54,  -0.52 ,-0.5 , -0.48 ,
                          -0.46, -0.44 ,-0.42, -0.4 , -0.38 , -0.36,  -0.34 ,-0.32, -0.3 ,
                          -0.28, -0.26 ,-0.24, -0.22, -0.2  ,-0.18 , -0.16  ,-0.14 , -0.12 ,
                          -0.1 , -0.08 ,-0.06, -0.04, -0.02 , 0.0  , 0.02   ,0.04  , 0.06  ,0.08 ,
                          0.1  , 0.12  ,0.14 , 0.16 , 0.18  ,0.2   ,0.22    ,0.24   ,0.26  ,0.28  ,0.3 ,
                          0.32 , 0.34  ,0.36 , 0.38 , 0.4   ,0.42  ,0.44    ,0.46   ,0.48  ,0.5   ,0.52 ,
                          0.54 , 0.56  ,0.58 , 0.6  , 0.62  ,0.64  ,0.66    ,0.68   ,0.7   ,0.72  ,0.74 ,
                          0.76 , 0.78  ,0.8  , 0.82 , 0.84  ,0.86  ,0.88    ,0.9    ,0.92  ,0.94  ,0.96 ,
                          0.98 , 1.0])
n_folds       = 10
alpha = -0.01
beta  = 0.3
mu    = 1.0 # 1e4 # 0.1 # 1.0
results_dir = "/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/results"

# -----------------------------
# Outputs
# -----------------------------
best_train_lam   = []
best_test_lam    = []
best_lambda_lam  = []
best_train_std = []   # <-- needed for correct error bars
best_test_std  = []   # <-- needed for correct error bars

# track missing files
missing_files = []

# -----------------------------
# Main sweep
# -----------------------------
for lam in lam_values:

    train_means = []
    train_stds  = []
    test_means  = []
    test_stds   = []

    for lambda_value in lambda_values:

        train_vals = []
        test_vals  = []

        for fold in range(1, n_folds + 1):

            fpath = (
                f"{results_dir}/"
                f"fold_results-alpha-{alpha:.2f}-beta-{beta:.2f}-lam-{lam:.2f}-mu-{mu:.2f}-lambda-ridge-{lambda_value:.1e}-fold-{fold}.txt"
            )

            # -----------------------------
            # Checkpoint: file exists?
            # -----------------------------
            if not os.path.exists(fpath):
                missing_files.append((lam, lambda_value, fold))
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
        print(f"⚠️  All NaNs for lam={lam:.2f}, skipping")
        best_lambda_lam.append(np.nan)
        best_test_lam.append(np.nan)
        best_train_lam.append(np.nan)
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
    best_lambda_lam.append(best_lambda)
    best_test_lam.append(best_test)
    best_train_lam.append(best_train)

    best_test_std.append(best_test_s)
    best_train_std.append(best_train_s)


# -----------------------------
# Convert to arrays (fix typo)
# -----------------------------
best_train_lam   = np.array(best_train_lam)
best_test_lam    = np.array(best_test_lam)
best_lambda_lam  = np.array(best_lambda_lam)

best_train_std = np.array(best_train_std)
best_test_std  = np.array(best_test_std)

# save the values
np.save(f"best_train_over_lam-mu-{mu:.2f}.npy",best_train_lam)
np.save(f"best_train_std-mu-{mu:.2f}.npy",best_train_std)
np.save(f"best_test_over_lam-mu-{mu:.2f}.npy",best_test_lam)
np.save(f"best_test_std-mu-{mu:.2f}.npy",best_test_std)
np.save(f"best_lambda_over_lam-mu-{mu:.2f}.npy",best_lambda_lam)

# -----------------------------
# Plot with correct error bars
# -----------------------------
# plt.figure(figsize=(16, 8))

# plt.errorbar(
#     lam_values,
#     best_test_lam,
#     yerr=best_test_std,
#     marker='o',
#     linewidth=3,
#     capsize=6,
#     label="Test Accuracy"
# )

# plt.errorbar(
#     lam_values,
#     best_train_lam,
#     yerr=best_train_std,
#     marker='s',
#     linewidth=3,
#     capsize=6,
#     label="Train Accuracy"
# )

# # Formatting
# plt.xlabel("Lambda", fontweight='bold', fontsize=20)
# plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.legend(fontsize=18)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()

# plt.savefig(f"lambda-optimization-10folds-errorbars-with-best-lambdas-mu-{mu:.0e}.png", dpi=300)
# plt.close()

plt.style.use('ggplot')
plt.figure(figsize=(16, 8))

# --- Test accuracy ---
plt.plot(
    lam_values,
    best_test_lam,
    marker='o',
    linewidth=3,
    label="Val. Accuracy"
)

plt.fill_between(
    lam_values,
    best_test_lam - best_test_std,
    best_test_lam + best_test_std,
    alpha=0.4
)

# --- Train accuracy ---
plt.plot(
    lam_values,
    best_train_lam,
    marker='s',
    linewidth=3,
    label="Train Accuracy"
)

plt.fill_between(
    lam_values,
    best_train_lam - best_train_std,
    best_train_lam + best_train_std,
    alpha=0.4
)

# Baseline (without reservoir)
baseline_mean = 80.73327
baseline_std  = 1.2364021603426592
baseline_vals = np.full_like(lam_values, baseline_mean)
plt.plot(lam_values, baseline_vals, marker='o', label="Val. Acc. without Reservoir", color="gray")
plt.fill_between(lam_values, baseline_vals - baseline_std, baseline_vals + baseline_std, color="gray", alpha=0.2)

# --- Formatting ---
plt.xlabel("Lambda", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)

plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)

plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()

plt.savefig(f"lambda-optimization-10folds-fill-with-best-lambdas-mu-{mu:.0e}.png", dpi=300)
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
    for i, (lam, lambda_value, fold) in enumerate(missing_files[:max_print]):
        print(f"[{i+1:02d}] a={lam:.2f}, lambda={lambda_value:.1e}, fold={fold}")

    if len(missing_files) > max_print:
        print(f"... and {len(missing_files) - max_print} more")
else:
    print("\n✅ All result files loaded successfully.")
