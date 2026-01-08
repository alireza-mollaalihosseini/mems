import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# -----------------------------
# Configuration
# -----------------------------
n_folds = 10
results_dir = "/scratch/almo2783/scratch/ml-paper/lda/sensor/results"

# -----------------------------
# Load fold data per lambda
# -----------------------------
train_means = []
train_stds  = []
test_means  = []
test_stds   = []


train_vals = []
test_vals  = []

for fold in range(1, n_folds + 1):

    fpath = f"{results_dir}/fold_results-fold-{fold}-fft.txt"
    result = np.loadtxt(fpath)

    train_vals.append(result[0] * 100.0)
    test_vals.append(result[1] * 100.0)

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
best_test   = test_means[best_idx]
best_train  = train_means[best_idx]
best_test_std = test_stds[best_idx]
best_train_std = train_stds[best_idx]

print("The best parameters and values:\n")
print(f"The best test value: {best_test}")
print(f"The best train value: {best_train}")
print(f"The best test std value: {best_test_std}")
print(f"The best train std value: {best_train_std}")