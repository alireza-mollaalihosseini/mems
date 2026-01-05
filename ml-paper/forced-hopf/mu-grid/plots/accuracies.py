import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
lambda_values = np.array([
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    1, 10, 1e2, 1e3, 1e4, 1e5, 1e6
])

mu_values = np.array([
    1e-2, 2e-2, 5e-2, 1e-1, 2.2e-1, 4.6e-1,
    1, 2.15, 4.64, 10, 21.5, 46.5,  
    1e2, 2.15e2, 4.64e2, 1e3, 2.15e3, 4.64e3,
    1e4, 2.15e4, 4.64e4, 1e5, 2.15e5, 6.64e5,
    1e6
])

lam   = 0.1
alpha = -0.01
beta  = 0.3

results_dir = "/scratch/almo2783/scratch/ml-paper/forced-hopf/mu-grid/results"

# -------------------------
# Containers
# -------------------------
best_acc_per_mu = np.zeros(len(mu_values))
best_train_acc_per_mu = np.zeros(len(mu_values))
best_lambda_per_mu = np.zeros(len(mu_values))

# -------------------------
# Load & analyze results
# -------------------------
for i, mu in enumerate(mu_values):

    fname = os.path.join(
        results_dir,
        f"results-mu-{mu:.2e}-lam-{lam}-alpha-{alpha}-beta-{beta}.txt"
    )

    if not os.path.exists(fname):
        print(f"WARNING: Missing {fname}")
        best_acc_per_mu[i] = np.nan
        best_lambda_per_mu[i] = np.nan
        continue

    # columns:
    # 0 → lambda
    # 1 → train accuracy
    # 2 → validation/test accuracy
    data = np.loadtxt(fname, skiprows=1)

    lambdas = data[:, 0]
    train_acc = data[:, 1]
    test_acc = data[:, 2]     # val_acc column

    # best lambda index
    idx_best = np.argmax(test_acc)

    best_acc_per_mu[i] = test_acc[idx_best]
    best_train_acc_per_mu[i] = train_acc[idx_best]
    best_lambda_per_mu[i] = lambdas[idx_best]

    print(f"mu = {mu:.1e} | best λ = {lambdas[idx_best]:.1e} | best acc = {test_acc[idx_best]:.4f}")


# -------------------------
# Global maximum (over μ)
# -------------------------
global_best_idx = np.nanargmax(best_acc_per_mu)

global_best_mu     = mu_values[global_best_idx]
global_best_acc    = best_acc_per_mu[global_best_idx]
global_best_lambda = best_lambda_per_mu[global_best_idx]

print("\n====== GLOBAL BEST RESULT ======")
print(f"Best μ       = {global_best_mu:.2e}")
print(f"Best λ       = {global_best_lambda:.2e}")
print(f"Best Val Acc = {global_best_acc:.4f}")
print("================================")

# -------------------------
# Plot
# -------------------------
plt.style.use('ggplot')
plt.figure(figsize=(16,8))
plt.semilogx(mu_values, best_acc_per_mu * 100, marker='o', label="Validation Acc.")
plt.semilogx(mu_values, best_train_acc_per_mu * 100, marker='o', label="Train Acc.")

# Highlight best point
plt.scatter(
    global_best_mu,
    global_best_acc * 100,
    s=120,
    marker='*',
    zorder=5,
    label=(
        f"Peak val acc = {global_best_acc * 100:.4f}\n"
        f"at μ = {global_best_mu:.1e}, λ = {global_best_lambda:.1e}"
    )
)

# Vertical line at best μ
plt.axvline(
    global_best_mu,
    linestyle=':',
    linewidth=1.5
)

plt.xlabel(r'$\mu$', fontweight='bold', fontsize=20)
plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=20)

plt.xticks(fontweight='bold', fontsize=16)
plt.yticks(fontweight='bold', fontsize=16)

plt.grid(True, which="both", linestyle='--', alpha=0.5)

plt.title(
    f"Best Validation Accuracy vs $\\mu$\n"
    f"(lam={lam}, alpha={alpha}, beta={beta})",
    fontweight='bold',
    fontsize=18
)
plt.grid(True, which='both', linestyle='--', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig("Val-opt-for-best-lambda.png", dpi=300)
plt.close()
