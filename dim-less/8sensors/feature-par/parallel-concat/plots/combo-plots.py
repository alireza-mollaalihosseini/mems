# import itertools
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# lambda_value = 1e4  # Only use lambda = 1e4
# top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 
#                 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]

# # Length of extracted features data
# feature_lengths = {
#     1e-6: {k: k for k in top_k_values}
# }

# # Define sensor combinations (example names – adapt to your naming convention!)
# combinations = []
# sensors = [1,2,3,4,5,6,7]
# # Generate only pairs, triplets, quartets, and all 7
# subsets = []
# subsets.extend(itertools.combinations(sensors, 2))  # pairs
# subsets.extend(itertools.combinations(sensors, 3))  # triplets
# # subsets.extend(itertools.combinations(sensors, 4))  # quartets
# subsets.append(tuple(sensors))

# for subset in subsets:
#   combinations.append("-".join(map(str, subset)))

# combinations = np.array(combinations)

# plt.figure(figsize=(16, 8))

# # Extract node values for x-axis
# nodes = [feature_lengths[1e-6].get(top_k, 0) for top_k in top_k_values]

# for combo in combinations:
#     val_accuracies = []

#     for top_k in top_k_values:
#         try:
#             # Adjust the path: assumes each combination has its own folder
#             data = np.loadtxt(
#                 f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/combo/"
#                 f"results_val-sensors-{combo}-topk-{top_k}-lambda-{lambda_value}.txt"
#             )
#             val_accuracy = data[2] * 100  # Convert to percentage
#             val_accuracies.append(val_accuracy)
#         except Exception:
#             val_accuracies.append(np.nan)

#     # Plot each combination
#     # plt.plot(nodes, val_accuracies, marker='o', linewidth=2, label=f"Val Acc. {combo}")
#     plt.plot(nodes, val_accuracies, marker='o', linewidth=2)

# # Customize the plot
# plt.xscale('log')  # Log scale for nodes
# plt.grid(True, linestyle='--', alpha=0.5)
# # plt.legend(loc='best', fontsize=14)
# plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel("Top-k Features", fontsize=16, fontweight='bold')
# plt.ylabel("Validation Accuracy (%)", fontsize=16, fontweight='bold')
# plt.title("Validation Accuracy across Sensor Combinations", fontsize=18, fontweight='bold')
# plt.tight_layout()

# # Save the figure
# plt.savefig(f"accuracy_vs_nodes_combinations_lambda_1e4_val.png", bbox_inches='tight', dpi=300)
# plt.close()



# import itertools
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# lambda_value = 1e4  # Only use lambda = 1e4
# top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 
#                 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]

# # Length of extracted features data
# feature_lengths = {
#     1e-6: {k: k for k in top_k_values}
# }

# # Define sensor combinations (example names – adapt to your naming convention!)
# combinations = []
# sensors = [1,2,3,4,5,6,7]
# # Generate only pairs, triplets, quartets, and all 7
# subsets = []
# subsets.extend(itertools.combinations(sensors, 2))  # pairs
# subsets.extend(itertools.combinations(sensors, 3))  # triplets
# # subsets.extend(itertools.combinations(sensors, 4))  # quartets
# subsets.append(tuple(sensors))

# for subset in subsets:
#   combinations.append("-".join(map(str, subset)))

# combinations = np.array(combinations)

# plt.figure(figsize=(16, 8))

# # Extract node values for x-axis
# nodes = [feature_lengths[1e-6].get(top_k, 0) for top_k in top_k_values]

# for combo in combinations:
#     test_accuracies = []

#     for top_k in top_k_values:
#         try:
#             # Adjust the path: assumes each combination has its own folder
#             data = np.loadtxt(
#                 f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/combo/"
#                 f"results_test-sensors-{combo}-topk-{top_k}-lambda-{lambda_value}.txt"
#             )
#             test_accuracy = data[2] * 100  # Convert to percentage
#             test_accuracies.append(test_accuracy)
#         except Exception:
#             test_accuracies.append(np.nan)

#     # Plot each combination
#     # plt.plot(nodes, test_accuracies, marker='o', linewidth=2, label=f"Test Acc. {combo}")
#     plt.plot(nodes, test_accuracies, marker='o', linewidth=2)

# # Customize the plot
# plt.xscale('log')  # Log scale for nodes
# plt.grid(True, linestyle='--', alpha=0.5)
# # plt.legend(loc='best', fontsize=14)
# plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel("Top-k Features", fontsize=16, fontweight='bold')
# plt.ylabel("Test Accuracy (%)", fontsize=16, fontweight='bold')
# plt.title("Test Accuracy across Sensor Combinations", fontsize=18, fontweight='bold')
# plt.tight_layout()

# # Save the figure
# plt.savefig(f"accuracy_vs_nodes_combinations_lambda_1e4_test.png", bbox_inches='tight', dpi=300)
# plt.close()


import itertools
import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_value = 1e4
top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]

# Length of extracted features data
feature_lengths = {1e-6: {k: k for k in top_k_values}}

# Define sensor combinations
sensors = [1, 2, 3, 4, 5, 6, 7]
subsets = []
subsets.extend(itertools.combinations(sensors, 2))  # pairs
subsets.extend(itertools.combinations(sensors, 3))  # triplets
subsets.append(tuple(sensors))  # all 7

combinations = ["-".join(map(str, subset)) for subset in subsets]

plt.figure(figsize=(16, 8))

# Extract node values for x-axis
nodes = [feature_lengths[1e-6].get(top_k, 0) for top_k in top_k_values]

# Assign fixed colors per sensor count
color_map = {2: "tab:blue", 3: "tab:green", 7: "tab:red"}

# Track which labels were already plotted
plotted_labels = set()

for combo in combinations:
    val_accuracies = []

    for top_k in top_k_values:
        try:
            data = np.loadtxt(
                f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/parallel-concat/results/combo/"
                f"results_val-sensors-{combo}-topk-{top_k}-lambda-{lambda_value}.txt"
            )
            val_accuracy = data[2] * 100  # percentage
            val_accuracies.append(val_accuracy)
        except Exception:
            val_accuracies.append(np.nan)

    # Count sensors in this combo
    num_sensors = len(combo.split("-"))
    color = color_map.get(num_sensors, "gray")

    # Only label once per group
    if num_sensors not in plotted_labels:
        plt.plot(nodes, val_accuracies, marker='o', linewidth=2, color=color,
                 label=f"{num_sensors} sensors")
        plotted_labels.add(num_sensors)
    else:
        plt.plot(nodes, val_accuracies, linewidth=1, color=color, alpha=0.4)

# Customize the plot
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best', fontsize=14)
plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel("Top-k Features", fontsize=16, fontweight='bold')
plt.ylabel("Validation Accuracy (%)", fontsize=16, fontweight='bold')
plt.title("Validation Accuracy across Sensor Combinations", fontsize=18, fontweight='bold')
plt.tight_layout()

plt.savefig("accuracy_vs_nodes_combinations_lambda_1e4_val_with_labels.png", bbox_inches='tight', dpi=300)
plt.close()

