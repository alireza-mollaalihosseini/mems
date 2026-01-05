import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import coherence

def compute_coherences(sim_dir, a_values, fs=1e6):
    """
    Computes average coherence between the original signal and each simulated u_ac
    stored in the given directory using time series data.
    """
    # Load the reference/original signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    original_signal = original_data["signal"]

    coherences = []

    for a in a_values:
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a = {a:.2f}")
            coherences.append(np.nan)
            continue

        sim_data = np.load(sim_path)
        u_ac = sim_data["u_ac"]

        # Ensure signals are same length
        min_len = min(len(original_signal), len(u_ac))
        orig_trunc = original_signal[:min_len]
        uac_trunc = u_ac[:min_len]

        if min_len < 2:
            coherences.append(np.nan)
            continue

        # Compute coherence
        f, Cxy = coherence(orig_trunc, uac_trunc, fs=fs, nperseg=min(1024, min_len//4))
        avg_coherence = np.mean(Cxy)
        coherences.append(avg_coherence)

    return np.array(coherences)


# ------------------- RUN -------------------
if __name__ == "__main__":
    sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    save_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/plots"
    u_dc = 0.4
    a_values = np.linspace(-2, 2, 101)

    coherences = compute_coherences(sim_dir, a_values)

    # Save coherences
    np.save(os.path.join(save_dir, "coherences_avg.npy"), coherences)
    # print(f"✅ Coherences saved to {save_dir}/coherences_avg.npy")

    # Load validation and test data
    data_val  = np.load("/scratch/almo2783/scratch/ml-paper/filter-like-7/plots/data_matrix_val.npy")
    data_test = np.load("/scratch/almo2783/scratch/ml-paper/filter-like-7/plots/data_matrix_test.npy")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, coherences, marker="o", linewidth=2, label="Coherence (Avg)")
    plt.plot(a_values, data_val / 100, marker="o", linewidth=2, label="Validation")
    plt.plot(a_values, data_test / 100, marker="o", linewidth=2, label="Test")
    plt.axvline(x=0.44, color="r", linestyle="--", alpha=0.7)

    plt.title(f"Coherence vs a (u_dc = {u_dc})", fontsize=20, fontweight="bold")
    plt.xlabel("a values", fontsize=20, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight="bold", fontsize=18)
    plt.yticks(fontweight="bold", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"coherence_accuracy_vs_a_udc_{u_dc}_avg.png")
    plt.savefig(save_path, dpi=300)
    # print(f"✅ Plot saved: {save_path}")

    plt.close()