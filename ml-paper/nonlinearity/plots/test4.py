import numpy as np
import matplotlib.pyplot as plt
import os

def compute_correlations(sim_dir, a_values):
    """
    Computes correlation between the original signal and each simulated u_ac
    stored in the given directory using FFT magnitudes.
    """
    # Load the reference/original signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    original_signal = original_data["signal"]
    fft_original = np.abs(np.fft.rfft(original_signal))

    correlations = []

    for a in a_values:
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            correlations.append(np.nan)
            continue

        sim_data = np.load(sim_path)
        u_ac = sim_data["u_ac"]
        fft_u_ac = np.abs(np.fft.rfft(u_ac))

        # Match lengths
        N = min(len(fft_original), len(fft_u_ac))
        if N < 2:
            correlations.append(np.nan)
            continue

        corr = np.corrcoef(fft_original[:N], fft_u_ac[:N])[0, 1]
        correlations.append(corr)

    return np.array(correlations)


# ------------------- RUN -------------------
if __name__ == "__main__":
    sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
    u_dc = 0.4
    a_values = np.linspace(-1, 1, 101)

    correlations = compute_correlations(sim_dir, a_values)

    # Save correlations
    np.save(os.path.join(save_dir, "correlations_fft_new.npy"), correlations)
    # print(f"✅ Correlations saved to {save_dir}/correlations_fft.npy")

    # Load validation and test data
    data_train  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/train_accuracies.npy")
    data_val  = np.load("/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/val_accuracies.npy")
    # data_test = np.load("/scratch/almo2783/scratch/ml-paper/filter-like-7/plots/data_matrix_test.npy")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, correlations, marker="o", linewidth=2, label="Correlation (FFT)")
    plt.plot(a_values, data_train, marker="o", linewidth=2, label="Train")
    plt.plot(a_values, data_val, marker="o", linewidth=2, label="Validation")
    # plt.plot(a_values, data_test, marker="o", linewidth=2, label="Test")
    plt.axvline(x=0.44, color="r", linestyle="--", alpha=0.7)
    plt.axhline(y=0.7850, color='red', linestyle='--', alpha=0.7)

    plt.title(f"Correlation vs a (u_dc = {u_dc})", fontsize=20, fontweight="bold")
    plt.xlabel("a values", fontsize=20, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight="bold", fontsize=18)
    plt.yticks(fontweight="bold", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"correlation_accuracy_vs_a_udc_{u_dc}_fft_new.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
