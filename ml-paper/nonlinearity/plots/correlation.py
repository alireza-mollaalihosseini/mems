import numpy as np
import matplotlib.pyplot as plt
import os

def compute_correlations(sim_dir, a_values):
    """
    Computes correlation between the original signal and each simulated u_ac
    stored in the given directory.
    """
    # Load the reference/original signal
    original_data = np.load(os.path.join(sim_dir, "original_signal.npz"))
    original_signal = original_data["signal"]

    correlations = []

    for a in a_values:
        sim_path = os.path.join(sim_dir, f"u_ac_a_{a:.2f}.npz")
        if not os.path.exists(sim_path):
            print(f"⚠️ Missing file for a={a:.2f}")
            correlations.append(np.nan)
            continue

        sim_data = np.load(sim_path)
        u_ac = sim_data["u_ac"]

        # Match lengths
        N = min(len(u_ac), len(original_signal))
        if N < 2:
            correlations.append(np.nan)
            continue

        # Compute Pearson correlation
        corr = np.corrcoef(original_signal[:N], u_ac[:N])[0, 1]
        correlations.append(corr)

    return np.array(correlations)


def plot_correlations(a_values, correlations, u_dc, output_dir=None):
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, correlations, marker="o", linewidth=2)
    plt.title(f"Correlation vs a (u_dc = {u_dc})", fontsize=20, fontweight='bold')
    plt.xlabel("a values", fontsize=20, fontweight='bold')
    plt.ylabel("Correlation", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"correlation_vs_a_udc_{u_dc}.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved: {save_path}")

    plt.show()


# ------------------- RUN -------------------
if __name__ == "__main__":
    sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results/"
    save_dir = '/scratch/almo2783/scratch/ml-paper/nonlinearity/plots'
    u_dc = 0.4
    a_values = np.linspace(-2, 2, 101)

    correlations = compute_correlations(sim_dir, a_values)
    # plot_correlations(a_values, correlations, u_dc, output_dir=sim_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(a_values, correlations, marker="o", linewidth=2)
    plt.axvline(x=0.44, color='r', linestyle='--', alpha=0.7)
    plt.title(f"Correlation vs a (u_dc = {u_dc})", fontsize=20, fontweight='bold')
    plt.xlabel("a values", fontsize=20, fontweight='bold')
    plt.ylabel("Correlation", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"correlation_vs_a_udc_{u_dc}.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved: {save_path}")

    plt.close()