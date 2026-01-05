import numpy as np
import matplotlib.pyplot as plt

a_values = np.linspace(-1,1, 101)

best_train_a_win_1   = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots-10fold/best_train_over_a.npy")
best_test_a_win_1    = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots-10fold/best_test_over_a.npy")
best_lambda_a_win_1  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots-10fold/best_lambda_over_a.npy")
best_train_std_win_1   = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots-10fold/best_train_std.npy")
best_test_std_win_1    = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots-10fold/best_test_std.npy")

n_windows = 3
best_train_a_win_3   = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_train_over_a-win-{int(n_windows)}.npy")
best_test_a_win_3    = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_test_over_a-win-{int(n_windows)}.npy")
best_lambda_a_win_3  = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_lambda_over_a-win-{int(n_windows)}.npy")
best_train_std_win_3   = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_train_std-win-{int(n_windows)}.npy")
best_test_std_win_3    = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_test_std-win-{int(n_windows)}.npy")

n_windows = 10
best_train_a_win_10   = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_train_over_a-win-{int(n_windows)}.npy")
best_test_a_win_10    = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_test_over_a-win-{int(n_windows)}.npy")
best_lambda_a_win_10  = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_lambda_over_a-win-{int(n_windows)}.npy")
best_train_std_win_10   = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_train_std-win-{int(n_windows)}.npy")
best_test_std_win_10    = np.load(f"/scratch/almo2783/scratch/ml-paper/windows/plots/best_test_std-win-{int(n_windows)}.npy")



# plt.style.use('ggplot')
# plt.figure(figsize=(16,8))
# plt.errorbar(a_values, best_test_a_win_1, yerr=best_test_std_win_1, marker='o', linewidth=3, capsize=6, label="Val. Accuracy ($win=1$)")
# plt.errorbar(a_values, best_test_a_win_3, yerr=best_test_std_win_3, marker='o', linewidth=3, capsize=6, label="Val. Accuracy ($win=3$)")
# plt.errorbar(a_values, best_test_a_win_10, yerr=best_test_std_win_10, marker='o', linewidth=3, capsize=6, label="Val. Accuracy ($win=10$)")
# plt.errorbar(a_values, [80.73327] * len(a_values), yerr=1.2364021603426592, marker="o", label="Val. Acc. without Reservoir")
# plt.xlabel("Feedback (a)", fontweight='bold', fontsize=20)
# plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
# plt.title("Accuracy for various windows over Feedback ($u_{dc}=0.4, \\mu=1.0$)", fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.legend(fontsize=18)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()
# plt.savefig("comparison-test.png", dpi=300)
# plt.close()

plt.style.use('ggplot')
plt.figure(figsize=(16,8))

# win = 1
plt.plot(a_values, best_test_a_win_1, marker='o', linewidth=3, label="Val. Accuracy ($win=1$)")
plt.fill_between(a_values, best_test_a_win_1 - best_test_std_win_1, best_test_a_win_1 + best_test_std_win_1, alpha=0.4)

# win = 3
plt.plot(a_values, best_test_a_win_3, marker='o', linewidth=3, label="Val. Accuracy ($win=3$)")
plt.fill_between(a_values, best_test_a_win_3 - best_test_std_win_3, best_test_a_win_3 + best_test_std_win_3, alpha=0.4)

# win = 10
plt.plot(a_values, best_test_a_win_10, marker='o', linewidth=3, label="Val. Accuracy ($win=10$)")
plt.fill_between(a_values, best_test_a_win_10 - best_test_std_win_10, best_test_a_win_10 + best_test_std_win_10, alpha=0.4)

# Baseline (without reservoir)
baseline_mean = 80.73327
baseline_std  = 1.2364021603426592
baseline_vals = np.full_like(a_values, baseline_mean)
plt.plot(a_values, baseline_vals, marker='o', label="Val. Acc. without Reservoir")
plt.fill_between(a_values, baseline_vals - baseline_std, baseline_vals + baseline_std, alpha=0.2)

# Labels & formatting
plt.xlabel("Feedback (a)", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
plt.title("Accuracy for various windows over Feedback ($u_{dc}=0.4, \\mu=1.0$)", fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("comparison-test.png", dpi=300)
plt.close()




plt.figure(figsize=(16,8))
plt.plot(a_values, best_lambda_a_win_1, marker='o', linewidth=3, label="Reg. ($win=1$)")
plt.plot(a_values, best_lambda_a_win_3, marker='o', linewidth=3, label="Reg. ($win=3$)")
plt.plot(a_values, best_lambda_a_win_10, marker='o', linewidth=3, label="Reg. ($win=10$)")
plt.xlabel("Feedback (a)", fontweight='bold', fontsize=20)
plt.ylabel("Regularization", fontweight='bold', fontsize=20)
plt.yscale("log")
plt.title("Best regularization value for various windows over Feedback ($u_{dc}=0.4, \\mu=1.0$)", fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("comparison-lambda.png", dpi=300)
plt.close()