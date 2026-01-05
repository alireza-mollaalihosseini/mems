import numpy as np
import matplotlib.pyplot as plt

lam_values = np.linspace(-1,1, 101)

mu = 1.0
best_train_lam_mu_1   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_over_lam-mu-{mu:.2f}.npy")
best_test_lam_mu_1    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_over_lam-mu-{mu:.2f}.npy")
best_lambda_lam_mu_1  = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_lambda_over_lam-mu-{mu:.2f}.npy")
best_train_std_mu_1   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_std-mu-{mu:.2f}.npy")
best_test_std_mu_1    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_std-mu-{mu:.2f}.npy")

mu = 0.1
best_train_lam_mu_01   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_over_lam-mu-{mu:.2f}.npy")
best_test_lam_mu_01    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_over_lam-mu-{mu:.2f}.npy")
best_lambda_lam_mu_01  = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_lambda_over_lam-mu-{mu:.2f}.npy")
best_train_std_mu_01   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_std-mu-{mu:.2f}.npy")
best_test_std_mu_01    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_std-mu-{mu:.2f}.npy")

mu = 1e4
best_train_lam_mu_1e4   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_over_lam-mu-{mu:.2f}.npy")
best_test_lam_mu_1e4    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_over_lam-mu-{mu:.2f}.npy")
best_lambda_lam_mu_1e4  = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_lambda_over_lam-mu-{mu:.2f}.npy")
best_train_std_mu_1e4   = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_train_std-mu-{mu:.2f}.npy")
best_test_std_mu_1e4    = np.load(f"/scratch/almo2783/scratch/ml-paper/forced-hopf/10fold/plots/best_test_std-mu-{mu:.2f}.npy")



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
plt.plot(lam_values, best_test_lam_mu_1, marker='o', linewidth=3, label="Val. Accuracy ($\\mu=1.0$)")
plt.fill_between(lam_values, best_test_lam_mu_1 - best_test_std_mu_1, best_test_lam_mu_1 + best_test_std_mu_1, alpha=0.4)

# win = 3
plt.plot(lam_values, best_test_lam_mu_01, marker='o', linewidth=3, label="Val. Accuracy ($\\mu=0.1$)")
plt.fill_between(lam_values, best_test_lam_mu_01 - best_test_std_mu_01, best_test_lam_mu_01 + best_test_std_mu_01, alpha=0.4)

# win = 10
plt.plot(lam_values, best_test_lam_mu_1e4, marker='o', linewidth=3, label="Val. Accuracy ($\\mu=1e4$)")
plt.fill_between(lam_values, best_test_lam_mu_1e4 - best_test_std_mu_1e4, best_test_lam_mu_1e4 + best_test_std_mu_1e4, alpha=0.4)

# Baseline (without reservoir)
baseline_mean = 80.73327
baseline_std  = 1.2364021603426592
baseline_vals = np.full_like(lam_values, baseline_mean)
plt.plot(lam_values, baseline_vals, marker='o', label="Val. Acc. without Reservoir")
plt.fill_between(lam_values, baseline_vals - baseline_std, baseline_vals + baseline_std, alpha=0.2)

# Labels & formatting
plt.xlabel("$\\lambda$", fontweight='bold', fontsize=20)
plt.ylabel("Accuracy (%)", fontweight='bold', fontsize=20)
plt.title("Accuracy for various $\\mu$ values over $\\lambda$ for simplified Hopf ($\\alpha=-0.01, \\beta=0.3$)", fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("comparison-test.png", dpi=300)
plt.close()




plt.figure(figsize=(16,8))
plt.plot(lam_values, best_lambda_lam_mu_1, marker='o', linewidth=3, label="Reg. ($\\mu=1.0$)")
plt.plot(lam_values, best_lambda_lam_mu_01, marker='o', linewidth=3, label="Reg. ($\\mu=0.1$)")
plt.plot(lam_values, best_lambda_lam_mu_1e4, marker='o', linewidth=3, label="Reg. ($\\mu=1e4$)")
plt.xlabel("$\\lambda$", fontweight='bold', fontsize=20)
plt.ylabel("Regularization", fontweight='bold', fontsize=20)
plt.yscale("log")
plt.title("Best regularization value for various $\\mu$ values over $\\lambda$ for simplified Hopf ($\\alpha=-0.01, \\beta=0.3$)", fontsize=20)
plt.xticks(fontweight='bold', fontsize=20)
plt.yticks(fontweight='bold', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("comparison-reg.png", dpi=300)
plt.close()