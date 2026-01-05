import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# -----------------------------
# Configuration
# -----------------------------
lambda_values = np.array([
    1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
    1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
    1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18
])
a_values = [-2, 2]
u_dc_values = [0.1, 0.5]
results_dir = "/scratch/almo2783/scratch/ml-paper/multi-sens/accuracy"

for a in a_values:
    for u_dc in u_dc_values:


        fname = os.path.join(
            results_dir,
            f"results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-1.00e+00.txt"
        )

        if not os.path.exists(fname):
            print(f"WARNING: Missing {fname}")

        data = np.loadtxt(fname, skiprows=1)

        lambdas   = data[:, 0]
        train_acc = data[:, 1] * 100
        test_acc  = data[:, 2] * 100

        idx_best = np.argmax(test_acc)

        best_test   = test_acc[idx_best]
        best_train  = train_acc[idx_best]
        best_lambda = lambdas[idx_best]

        # std across λ (for uncertainty band)
        std_test = np.std(test_acc)
        std_train = np.std(train_acc)

        # -----------------------------
        # Best lambda from test mean
        # -----------------------------

        print("The best parameters and values:\n")
        print(f"The best lambda value: {best_lambda}")
        print(f"The best test value: {best_test}")
        print(f"The best train value: {best_train}")
        print(f"The best test std value: {std_test}")
        print(f"The best train std value: {std_train}")

        # -----------------------------
        # Plot with error bars
        # -----------------------------
        plt.figure(figsize=(16, 8))

        plt.plot(
            lambda_values, test_acc,
            marker='o',
            linewidth=3,
            label="Validation Accuracy"
        )

        plt.plot(
            lambda_values, train_acc,
            marker='s',
            linewidth=3,
            linestyle='--',
            label="Training Accuracy"
        )

        # highlight best lambda
        plt.scatter(
            best_lambda,
            best_test,
            s=200,
            marker='*',
            zorder=5,
            label=(
                f"Best val = {best_test:.2f}%\n"
                f"λ = {best_lambda:.1e}"
            )
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

        plt.savefig(f"lambda-optimization-a-{a:.2f}-u_dc-{u_dc:.2f}.png", dpi=300)
        plt.close()

        print(f"\nBest λ = {best_lambda:.1e} → Test Accuracy = {best_test:.2f}%")