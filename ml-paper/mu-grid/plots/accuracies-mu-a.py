# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # -------------------------
# # Configuration
# # -------------------------
# lambda_values = np.array([
#     1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
#     1, 10, 1e2, 1e3, 1e4, 1e5, 1e6
# ])

# mu_values = np.array([
#     1e-18, 1e-17, 1e-16, 1e-15, 1e-14,
#     1e-13, 1e-12, 1e-11, 1e-10, 1e-9,
#     1e-8,  1e-7,  1e-6,  1e-5,  1e-4,
#     1e-3,  1e-2,  1e-1,  1,     10,
#     1e2,   1e3,   1e4,   1e5,   1e6
# ])

# # NEW: sweep over a
# a_values = np.linspace(-1.0, 1.0, 101)

# u_dc = 0.4

# results_dir = "/scratch/almo2783/scratch/ml-paper/mu-grid/results"

# # -------------------------
# # Containers
# # -------------------------
# # shape: (n_a, n_mu)
# best_acc = np.full((len(a_values), len(mu_values)), np.nan)
# best_train_acc = np.full_like(best_acc, np.nan)
# best_lambda = np.full_like(best_acc, np.nan)

# # -------------------------
# # Load & analyze results
# # -------------------------
# for ia, a in enumerate(a_values):

#     for im, mu in enumerate(mu_values):

#         fname = os.path.join(
#             results_dir,
#             f"results-a-{a:.2f}-mu-{mu:.2e}.txt"
#         )

#         if not os.path.exists(fname):
#             print(f"WARNING: Missing {fname}")
#             continue

#         data = np.loadtxt(fname, skiprows=1)

#         lambdas = data[:,0]
#         train_acc = data[:,1]
#         test_acc  = data[:,2]

#         idx_best = np.argmax(test_acc)

#         best_acc[ia, im]       = test_acc[idx_best]
#         best_train_acc[ia,im] = train_acc[idx_best]
#         best_lambda[ia, im]   = lambdas[idx_best]


# # -------------------------
# # Best μ for each a
# # -------------------------
# best_mu_idx_per_a = np.nanargmax(best_acc, axis=1)

# best_mu_per_a      = mu_values[best_mu_idx_per_a]
# best_acc_per_a     = best_acc[np.arange(len(a_values)), best_mu_idx_per_a]
# best_lambda_per_a = best_lambda[np.arange(len(a_values)), best_mu_idx_per_a]

# # -------------------------
# # Global maximum
# # -------------------------
# global_idx     = np.nanargmax(best_acc)
# ia_best, im_best = np.unravel_index(global_idx, best_acc.shape)

# global_best_a      = a_values[ia_best]
# global_best_mu     = mu_values[im_best]
# global_best_lambda = best_lambda[ia_best, im_best]
# global_best_acc    = best_acc[ia_best, im_best]

# print("\n====== GLOBAL BEST RESULT ======")
# print(f"Best a       = {global_best_a:.3f}")
# print(f"Best μ       = {global_best_mu:.2e}")
# print(f"Best λ       = {global_best_lambda:.2e}")
# print(f"Best Val Acc = {global_best_acc:.4f}")
# print("================================")


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
    1e-18, 1e-17, 1e-16, 1e-15, 1e-14,
    1e-13, 1e-12, 1e-11, 1e-10, 1e-9,
    1e-8,  1e-7,  1e-6,  1e-5,  1e-4,
    1e-3,  1e-2,  1e-1,  1,     10,
    1e2,   1e3,   1e4,   1e5,   1e6
])

a_values = np.linspace(-1.0, 1.0, 101)
u_dc = 0.4
results_dir = "/scratch/almo2783/scratch/ml-paper/mu-grid/results"

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
        im = np.where(np.isclose(mu_values, mu_val))[0][0]
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


# # -------------------------
# # 2D Acc Heatmap (a vs μ)
# # -------------------------
# plt.style.use('ggplot')
# plt.figure(figsize=(18, 8))

# plt.imshow(
#     best_acc * 100,
#     aspect='auto',
#     origin='lower',
# )

# plt.colorbar(label="Validation Accuracy (%)")

# plt.yticks(
#     np.linspace(0, len(a_values) - 1, 9),
#     np.round(np.linspace(a_values.min(), a_values.max(), 9), 2)
# )
# plt.xticks(
#     np.arange(len(mu_values)),
#     [f"{mu:.0e}" for mu in mu_values],
#     rotation=45
# )

# plt.xlabel(r'$\mu$', fontweight='bold', fontsize=16)
# plt.ylabel('a', fontweight='bold', fontsize=16)

# plt.title(
#     f"Validation Accuracy Heatmap\n"
#     f"(u_dc={u_dc})",
#     fontweight='bold',
#     fontsize=18
# )

# plt.tight_layout()
# plt.savefig("heatmap_val_acc_a_mu.png", dpi=300)
# plt.close()

# -------------------------
# 2D Acc Heatmap (a vs μ) using pcolormesh
# -------------------------

plt.style.use('ggplot')
plt.figure(figsize=(18,8))

# Create coordinate grid
MU, A = np.meshgrid(mu_values, a_values)

pcm = plt.pcolormesh(
    MU,
    A,
    best_acc * 100,
    shading="auto"
)

plt.xscale("log")

cbar = plt.colorbar(pcm)
cbar.set_label("Validation Accuracy (%)", fontweight='bold', fontsize=16)
cbar.ax.tick_params(labelsize=14)

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
