import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# -------------------------
# Parameter grids
# -------------------------
a_values = np.linspace(-2, 2, 41)
u_dc_values = np.linspace(0.1, 1.0, 10)

path = "/scratch/almo2783/scratch/ml-paper/multi-sens/accuracy"

# -------------------------
# Storage
# -------------------------
results = {}
missing = []

# -------------------------
# Load results
# -------------------------
for a in a_values:
    for u_dc in u_dc_values:
        fname = os.path.join(
            path, f"results-a-{a:.2f}-u_dc-{u_dc:.2f}-mu-1.00e+00.txt"
        )

        if not os.path.exists(fname):
            missing.append((a, u_dc))
            continue

        try:
            data = np.loadtxt(fname, skiprows=1)
            if data.ndim == 1 or data.shape[1] < 3:
                raise ValueError("Invalid data shape")

            train_acc = data[:, 1].max() * 100.0
            val_acc = data[:, 2].max() * 100.0
            best_lambda = data[:, 0][np.argmax(data[:, 2])]

            results[(a, u_dc)] = (train_acc, val_acc, best_lambda)

        except Exception as e:
            print(f"Failed to read {fname}: {e}")
            missing.append((a, u_dc))

# -------------------------
# Report missing results
# -------------------------
for a, u_dc in missing:
    print(f"Missing results for a={a:.2f}, u_dc={u_dc:.2f}")


# print the best validation accuracy
best_val = -np.inf
best_params = None
for (a, u_dc), (tr, va, bl) in results.items():
    if va > best_val:
        best_val = va
        best_params = (a, u_dc, bl)

print(f"Best validation accuracy: {best_val:.2f}% at a={best_params[0]:.2f}, u_dc={best_params[1]:.2f}, lambda={best_params[2]:.2e}")

# -------------------------
# Prepare grids (NaN-safe)
# -------------------------
train_acc_grid = np.full((len(a_values), len(u_dc_values)), np.nan)
val_acc_grid = np.full_like(train_acc_grid, np.nan)
best_lambda_grid = np.full_like(train_acc_grid, np.nan)

for i, a in enumerate(a_values):
    for j, u_dc in enumerate(u_dc_values):
        if (a, u_dc) in results:
            train_acc_grid[i, j], val_acc_grid[i, j], best_lambda_grid[i, j] = results[(a, u_dc)]

# -------------------------
# Meshgrid
# -------------------------
U_dc_grid, A_grid = np.meshgrid(u_dc_values, a_values)

# -------------------------
# Plotting
# -------------------------
def plot_grid(Z, title, cbar_label, fname):
    plt.figure(figsize=(16, 8))
    plt.pcolormesh(U_dc_grid, A_grid, Z, shading="auto", cmap="viridis")
    plt.colorbar(label=cbar_label)
    plt.xlabel("u_dc", fontsize=20)
    plt.ylabel("a", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20)
    plt.savefig(fname, dpi=300)
    plt.close()

plot_grid(train_acc_grid, "Training Accuracy over Parameter Grid", "Training Accuracy (%)", "train_accuracy_grid.png")
plot_grid(val_acc_grid, "Validation Accuracy over Parameter Grid", "Validation Accuracy (%)", "val_accuracy_grid.png")
plot_grid(best_lambda_grid, "Best Lambda over Parameter Grid", "Best Lambda", "best_lambda_grid.png")

# -------------------------
# Save numeric grids
# -------------------------
np.save("train_accuracy_grid.npy", train_acc_grid)
np.save("val_accuracy_grid.npy", val_acc_grid)
np.save("best_lambda_grid.npy", best_lambda_grid)
np.save("missing_points.npy", np.array(missing))

# -------------------------
# Save full results table
# -------------------------
with open("results_grid.csv", "w") as f:
    f.write("a,u_dc,train_acc,val_acc,best_lambda\n")
    for (a, u_dc), (tr, va, bl) in results.items():
        f.write(f"{a},{u_dc},{tr},{va},{bl}\n")
