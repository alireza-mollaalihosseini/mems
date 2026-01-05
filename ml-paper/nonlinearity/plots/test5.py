import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from scipy import signal  # For potential future extensions, but not used here


def compute_nonlinearity(sim_dir, a_values, poly_degree=3):
    """
    Computes a nonlinearity metric for each simulated u_ac vs. original signal.
    Uses polynomial fitting: delta R^2 between quadratic and linear models.
    Higher values indicate stronger nonlinearity (improved fit from adding quadratic term).
    """
    # Load the reference/original signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    x = original_data["signal"]  # Input as predictor

    nonlinearities = []

    for a in a_values:
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f} (nonlinearity)")
            nonlinearities.append(np.nan)
            continue

        sim_data = np.load(sim_path)
        y = sim_data["u_ac"]  # Output as response

        # Match lengths
        N = min(len(x), len(y))
        if N < 10:  # Need sufficient points for fitting
            nonlinearities.append(np.nan)
            continue

        x_short = x[:N]
        y_short = y[:N]

        # Fit linear model (degree 1)
        coeffs_lin = np.polyfit(x_short, y_short, 1)
        y_pred_lin = np.polyval(coeffs_lin, x_short)
        r2_lin = r2_score(y_short, y_pred_lin)

        # Fit quadratic model (degree 2)
        coeffs_quad = np.polyfit(x_short, y_short, poly_degree)
        y_pred_quad = np.polyval(coeffs_quad, x_short)
        r2_quad = r2_score(y_short, y_pred_quad)

        # Nonlinearity metric: improvement from quadratic term (normalized)
        # Avoid division by zero; if linear already perfect, nonlinearity=0
        if (1 - r2_lin) > 1e-10:
            delta_r2 = (r2_quad - r2_lin) / (1 - r2_lin)
        else:
            delta_r2 = 0.0

        nonlinearities.append(delta_r2)

    return np.array(nonlinearities)


# ------------------- RUN -------------------
if __name__ == "__main__":
    sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
    u_dc = 0.4
    a_values = np.linspace(-1, 1, 101)

    correlations = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/correlations_fft_new.npy")
    nonlinearities_3 = compute_nonlinearity(sim_dir, a_values)

    # Save correlations and nonlinearities
    np.save(os.path.join(save_dir, "nonlinearities_delta_r2_3.npy"), nonlinearities_3)

    # Load validation and test data
    data_train  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/train_accuracies.npy")
    data_val  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/val_accuracies.npy")
    nonlinearities_2 = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/nonlinearities_delta_r2_2.npy")
    # data_test = np.load("/scratch/almo2783/scratch/ml-paper/filter-like-7/plots/data_matrix_test.npy")

    # normalize nonlinearities between 0 and 1
    nonlinearities_2 = (nonlinearities_2 - np.min(nonlinearities_2)) / (np.max(nonlinearities_2) - np.min(nonlinearities_2))
    nonlinearities_3 = (nonlinearities_3 - np.min(nonlinearities_3)) / (np.max(nonlinearities_3) - np.min(nonlinearities_3))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, correlations, marker="o", linewidth=2, label="Correlation (FFT)", color='blue')
    plt.plot(a_values, data_train, marker="o", linewidth=2, label="Train", color='orange')
    plt.plot(a_values, data_val, marker="o", linewidth=2, label="Validation", color='green')
    plt.plot(a_values, nonlinearities_2, marker="s", linewidth=2, label="Nonlinearity (ΔR²) second", color='red')
    plt.plot(a_values, nonlinearities_3, marker="s", linewidth=2, label="Nonlinearity (ΔR²) third", color='black')
    # plt.plot(a_values, data_test, marker="o", linewidth=2, label="Test")
    plt.axvline(x=0.44, color="r", linestyle="--", alpha=0.7)
    plt.axhline(y=0.7850, color='red', linestyle='--', alpha=0.7)

    plt.title(f"Correlation & Nonlinearity vs a (u_dc = {u_dc})", fontsize=20, fontweight="bold")
    plt.xlabel("a values", fontsize=20, fontweight="bold")
    plt.ylabel("Metric Value", fontsize=20, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight="bold", fontsize=18)
    plt.yticks(fontweight="bold", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"correlation_accuracy_nonlinearity_vs_a_udc_{u_dc}_fft.png")
    plt.savefig(save_path, dpi=300)
    plt.close()