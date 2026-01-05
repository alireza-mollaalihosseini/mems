import numpy as np
import os
import re
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
lambda_values = np.array([
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    1, 10, 1e2, 1e3, 1e4, 1e5, 1e6
])

mu_values = np.array([
    1e-13, 1e-12, 1e-11, 1e-10, 1e-9,
    1e-8,  1e-7,  1e-6,  1e-5,  1e-4,
    1e-3,  1e-2,  1e-1,  1,     10,
    1e2,   1e3,   1e4,   1e5,   1e6
])


mu_lookup = {
    float(f"{mu:.2e}"): i
    for i, mu in enumerate(mu_values)
}


a_values = np.linspace(-1.0, 1.0, 21)
u_dc = 0.4
results_dir = "/scratch/almo2783/scratch/ml-paper/mu-grid/results/highmem"

# -------------------------
# Prepare containers
# -------------------------
best_acc       = np.full((len(a_values), len(mu_values)), np.nan)
best_train_acc = np.full((len(a_values), len(mu_values)), np.nan)
best_lambda    = np.full((len(a_values), len(mu_values)), np.nan)

# -------------------------
# Compile regex for filenames
# -------------------------
pattern = re.compile(r"results-a-([+-]?\d+\.\d+)-mu-(\d+\.\d+e[+-]?\d+)\.txt")

# -------------------------
# Scan directory for .txt files
# -------------------------
existing_files = [
    f for f in os.listdir(results_dir)
    if f.endswith(".txt") and f.startswith("results-a")
]

print(f"Found {len(existing_files)} result files.")

# Track missing later
expected_pairs = {(float(f"{a:.2f}"), float(f"{mu:.2e}")) 
                  for a in a_values for mu in mu_values}
found_pairs = set()

# -------------------------
# Load each file that exists
# -------------------------
for fname in existing_files:
    match = pattern.match(fname)
    if not match:
        continue

    a_str, mu_str = match.groups()

    a_val = float(a_str)
    mu_val = float(mu_str)

    # Save that we found this
    found_pairs.add((a_val, mu_val))

    # Convert to indices
    try:
        ia = np.where(np.isclose(a_values, a_val))[0][0]
        # im = np.where(np.isclose(mu_values, mu_val))[0][0]
        mu_key = float(f"{mu_val:.2e}")
        if mu_key not in mu_lookup:
            print(f"WARNING: Parsed mu={mu_val:.3e} not in grid")
            continue

        im = mu_lookup[mu_key]
    except IndexError:
        print(f"WARNING: Parsed (a={a_val}, mu={mu_val}) not in provided grids.")
        continue

    path = os.path.join(results_dir, fname)

    try:
        data = np.loadtxt(path, skiprows=1)

        lambdas   = data[:, 0]
        train_acc = data[:, 1]
        test_acc  = data[:, 2]

        idx_best = np.argmax(test_acc)

        best_acc[ia, im]       = test_acc[idx_best]
        best_train_acc[ia, im] = train_acc[idx_best]
        best_lambda[ia, im]    = lambdas[idx_best]

    except Exception as e:
        print(f"ERROR reading {fname}: {e}")
        continue

# -------------------------
# Compute missing combinations
# -------------------------
missing_pairs = sorted(list(expected_pairs - found_pairs))

print(f"\nTotal missing combinations: {len(missing_pairs)}")
for a_val, mu_val in missing_pairs[:20]:
    print(f"  missing: a={a_val:.2f}, mu={mu_val:.2e}")
if len(missing_pairs) > 20:
    print("  ...")

# Optional: save missing pairs for re-submission
missing_file = os.path.join(results_dir, "missing_jobs.txt")
with open(missing_file, "w") as f:
    for a_val, mu_val in missing_pairs:
        f.write(f"{a_val:.2f} {mu_val:.2e}\n")

print(f"\nMissing jobs saved to: {missing_file}")

# -------------------------
# Continue with your original analysis
# -------------------------
# best_mu_idx_per_a = np.nanargmax(best_acc, axis=1)
best_mu_idx_per_a = np.full(len(a_values), np.nan)

for ia in range(len(a_values)):
    row = best_acc[ia]
    if np.all(np.isnan(row)):
        print(f"WARNING: No valid μ results for a={a_values[ia]:.2f}")
        continue
    best_mu_idx_per_a[ia] = np.nanargmax(row)

valid = ~np.isnan(best_mu_idx_per_a)               # boolean mask
valid_idx = best_mu_idx_per_a[valid].astype(int)   # convert valid positions to int

# best_mu_per_a     = mu_values[best_mu_idx_per_a]
# best_acc_per_a    = best_acc[np.arange(len(a_values)), best_mu_idx_per_a]
# best_lambda_per_a = best_lambda[np.arange(len(a_values)), best_mu_idx_per_a]

# Prepare output arrays
best_mu_per_a      = np.full(len(a_values), np.nan)
best_acc_per_a     = np.full(len(a_values), np.nan)
best_lambda_per_a  = np.full(len(a_values), np.nan)

# Fill only where valid
best_mu_per_a[valid]      = mu_values[valid_idx]
best_acc_per_a[valid]     = best_acc[np.where(valid)[0], valid_idx]
best_lambda_per_a[valid]  = best_lambda[np.where(valid)[0], valid_idx]

global_idx = np.nanargmax(best_acc)
ia_best, im_best = np.unravel_index(global_idx, best_acc.shape)

global_best_a      = a_values[ia_best]
global_best_mu     = mu_values[im_best]
global_best_lambda = best_lambda[ia_best, im_best]
global_best_acc    = best_acc[ia_best, im_best]

print("\n=== GLOBAL BEST ===")
print(f"a       = {global_best_a:.3f}")
print(f"mu      = {global_best_mu:.3e}")
print(f"lambda  = {global_best_lambda:.3e}")
print(f"acc (%) = {global_best_acc:.3f}")


nan_cols = np.where(np.all(np.isnan(best_acc), axis=0))[0]
print("Empty μ columns:", nan_cols)
print("Corresponding μ values:", mu_values[nan_cols])



# # -------------------------
# # 2D Acc Heatmap (a vs μ) using pcolormesh
# # -------------------------

# plt.style.use('ggplot')
# plt.figure(figsize=(16,8))

# # Create coordinate grid
# MU, A = np.meshgrid(mu_values, a_values)

# pcm = plt.pcolormesh(
#     MU,
#     A,
#     best_acc * 100,
#     shading="nearest"
# )

# plt.xscale("log")

# cbar = plt.colorbar(pcm)
# cbar.set_label("Validation Accuracy (%)", fontweight='bold', fontsize=16)
# cbar.ax.tick_params(labelsize=14)

# plt.xlabel(r'$\mu$', fontweight='bold', fontsize=18)
# plt.ylabel('a', fontweight='bold', fontsize=18)

# plt.title(
#     f"Validation Accuracy Heatmap (a vs μ)\n(u_dc={u_dc})",
#     fontweight='bold',
#     fontsize=18
# )

# plt.tight_layout()
# plt.savefig("heatmap_val_acc_a_mu.png", dpi=300)
# plt.close()


# -------------------------
# 2D Acc Heatmap (a vs μ index)
# -------------------------

plt.style.use('ggplot')
plt.figure(figsize=(18, 8))

# Use μ indices instead of actual values
mu_idx = np.arange(len(mu_values))  # 0, 1, ..., 19
MU_IDX, A = np.meshgrid(mu_idx, a_values)

pcm = plt.pcolormesh(
    MU_IDX,
    A,
    best_acc * 100,
    shading="nearest"
)

cbar = plt.colorbar(pcm)
cbar.set_label("Validation Accuracy (%)", fontweight='bold', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# X ticks: show μ values as labels
plt.xticks(
    mu_idx,
    [f"{mu:.0e}" for mu in mu_values],
    rotation=45,
    fontsize=14,
    fontweight='bold'
)
plt.yticks(fontsize=14, fontweight='bold')

plt.xlabel(r'$\mu$', fontweight='bold', fontsize=18)
plt.ylabel('a', fontweight='bold', fontsize=18)

plt.title(
    f"Validation Accuracy Heatmap (a vs μ)\n(u_dc={u_dc})",
    fontweight='bold',
    fontsize=18
)

plt.tight_layout()
plt.savefig("heatmap_val_acc_a_mu.png", dpi=300)
plt.close()


# -------------------------
# Best Acc vs a
# -------------------------
plt.figure(figsize=(16,8))

plt.plot(a_values, best_acc_per_a * 100, linewidth=2)

plt.scatter(
    global_best_a,
    global_best_acc * 100,
    s=120,
    marker='*',
    zorder=5,
    label=(
        f"Peak acc = {global_best_acc*100:.2f}%\n"
        f"a={global_best_a:.2f}, μ={global_best_mu:.1e}, λ={global_best_lambda:.1e}"
    ),
)

plt.xlabel("a", fontweight='bold', fontsize=20)
plt.ylabel("Best Validation Accuracy (%)", fontweight='bold', fontsize=20)

plt.title(
    "Best Validation Accuracy vs a",
    fontweight='bold',
    fontsize=18
)

plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("best_val_acc_vs_a.png", dpi=300)
plt.close()
