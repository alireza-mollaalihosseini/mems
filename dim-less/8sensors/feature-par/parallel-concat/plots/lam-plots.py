# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import re
# import os

# # === Parameters ===
# top_k_values = [
#     10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
#     1000, 1500, 2000, 3000, 4000, 5000, 10000
# ]

# # Base results directory
# results_dir = "/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/lam-opt"

# # === Define sensor combinations ===
# sensors = [1, 2, 3, 4, 5, 6, 7]
# subsets = []
# subsets.extend(itertools.combinations(sensors, 2))  # pairs
# subsets.extend(itertools.combinations(sensors, 3))  # triplets
# subsets.append(tuple(sensors))  # all 7

# combinations = ["-".join(map(str, subset)) for subset in subsets]

# # === Plot setup ===
# plt.figure(figsize=(16, 8))
# nodes = top_k_values  # X-axis
# color_map = {2: "tab:blue", 3: "tab:green", 7: "tab:red"}
# plotted_labels = set()

# # === Loop over combinations ===
# for combo in combinations:
#     val_accuracies = []
#     num_sensors = len(combo.split("-"))
#     color = color_map.get(num_sensors, "gray")

#     for top_k in top_k_values:
#         # Search for any matching file (λ may vary)
#         pattern = os.path.join(
#             results_dir,
#             f"results_val-sensors-{combo}-topk-{top_k}-lambda-*.txt"
#         )
#         files = glob.glob(pattern)

#         if not files:
#             val_accuracies.append(np.nan)
#             continue

#         # If multiple matches exist, take the newest or first one
#         file_path = sorted(files)[-1]

#         try:
#             data = np.loadtxt(file_path)
#             val_accuracy = data[2] * 100  # percentage
#         except Exception:
#             val_accuracy = np.nan

#         val_accuracies.append(val_accuracy)

#     # === Plot ===
#     label = f"{num_sensors} sensors" if num_sensors not in plotted_labels else None
#     plt.plot(
#         nodes,
#         val_accuracies,
#         marker='o' if label else None,
#         linewidth=2 if label else 1,
#         color=color,
#         alpha=1.0 if label else 0.4,
#         label=label
#     )
#     plotted_labels.add(num_sensors)

# # === Style ===
# plt.xscale('log')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=14)
# plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel("Top-k Features", fontsize=16, fontweight='bold')
# plt.ylabel("Validation Accuracy (%)", fontsize=16, fontweight='bold')
# plt.title("Validation Accuracy (Optimal λ) across Sensor Combinations", fontsize=18, fontweight='bold')
# plt.tight_layout()

# # === Save ===
# plt.savefig("accuracy_vs_nodes_combinations_val_opt_lambda.png", bbox_inches='tight', dpi=300)
# plt.close()


import itertools
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# === Parameters ===
top_k_values = [
    10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1500, 2000, 3000, 4000, 5000, 10000
]

# Base results directory
results_dir = "/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/lam-opt"

# === Define sensor combinations ===
sensors = [1, 2, 3, 4, 5, 6, 7]
subsets = []
subsets.extend(itertools.combinations(sensors, 1))
subsets.extend(itertools.combinations(sensors, 2))  # pairs
subsets.extend(itertools.combinations(sensors, 3))  # triplets
subsets.extend(itertools.combinations(sensors, 4))  # quartets
subsets.extend(itertools.combinations(sensors, 5))  # quartets
subsets.extend(itertools.combinations(sensors, 6))  # quartets
subsets.append(tuple(sensors))  # all 7

combinations = ["-".join(map(str, subset)) for subset in subsets]

# === Plot setup ===
plt.figure(figsize=(16, 8))
nodes = top_k_values  # X-axis
# color_map = {2: "tab:blue", 3: "tab:green", 7: "tab:red"}
# plotted_labels = set()

# === Loop over combinations ===
for combo in combinations:
    val_accuracies = []
    num_sensors = len(combo.split("-"))
    # color = color_map.get(num_sensors, "gray")

    for top_k in top_k_values:
        # Search for any matching file (λ may vary)
        pattern = os.path.join(
            results_dir,
            f"results_val-sensors-{combo}-topk-{top_k}-lambda-*.txt"
        )
        files = glob.glob(pattern)

        if not files:
            val_accuracies.append(np.nan)
            continue

        # If multiple matches exist, take the newest or first one
        file_path = sorted(files)[-1]

        try:
            data = np.loadtxt(file_path)
            val_accuracy = data[2] * 100  # percentage
        except Exception:
            val_accuracy = np.nan

        val_accuracies.append(val_accuracy)

    # === Plot ===
    # label = f"{num_sensors} sensors" if num_sensors not in plotted_labels else None
    plt.plot(
        nodes,
        val_accuracies,
        marker='o',
        linewidth=2,
        # color=color,
        alpha=0.5
        # label=label
    )
    # plotted_labels.add(num_sensors)

# === Style ===
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=14)
plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Top-k Features", fontsize=20)
plt.ylabel("Validation Accuracy (%)", fontsize=20)
plt.title("Validation Accuracy (Optimal λ) across Sensor Combinations", fontsize=20)
plt.tight_layout()

# === Save ===
plt.savefig("accuracy_vs_nodes_combinations_val_opt_lambda-all.png", bbox_inches='tight', dpi=300)
plt.close()
