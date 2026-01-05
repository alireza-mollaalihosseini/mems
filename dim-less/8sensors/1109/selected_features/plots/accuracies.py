import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Parameters
a_value = 0.06
u_dc_value = 1.0
lambda_value = 1e4  # Only use lambda = 1e4
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]
alpha_values = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]
class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
                "street_traffic", "tram", "bus", "metro", "park"]

# Length of extracted features data
feature_lengths = {
    1e-6: {10: 10, 50: 50, 100: 100, 200: 200, 300: 300, 400: 400, 500: 500, 1000: 1000, 2000: 2000}
}

# Initialize data storage
test_accuracies = []
class_accuracies = {label: [] for label in class_labels}  # Store per-class accuracies

# Extract accuracy data for lambda = 1e4
for alpha in alpha_values:
    try:
        # Load test accuracy
        data = np.loadtxt(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/selected_features/results/results-a-{a_value}-alpha-{alpha}-lambda-{lambda_value}.txt")
        test_accuracy = data[4] * 100  # Convert to percentage
        test_accuracies.append(test_accuracy)

        # Load confusion matrix and extract per-class accuracies
        conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/selected_features/results/conf_matrix-a-{a_value}-alpha-{alpha}-lambda-{lambda_value}.txt")
        diags = conf_matrix.diagonal()
        class_totals = conf_matrix.sum(axis=1)

        for i, label in enumerate(class_labels):
            accuracy = (diags[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
            class_accuracies[label].append(accuracy)

    except Exception as e:
        test_accuracies.append(np.nan)
        for label in class_labels:
            class_accuracies[label].append(np.nan)  # Fill with NaNs if data is missing

# Extract node values for x-axis
nodes = [feature_lengths[1e-6].get(alpha, 0) for alpha in alpha_values]

# Plotting
# plt.style.use('dark_background')
plt.figure(figsize=(16, 8))

# Plot test accuracy in **black**
plt.plot(nodes, test_accuracies, marker='o', linewidth=2, color='black', label="Val. Acc.")

# Define colormap and assign unique colors to each class
cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ["teal", "darkorange"])
num_classes = len(class_labels)
colors = [cmap_custom(i / (num_classes - 1)) for i in range(num_classes)]  # Generate colors for each class

# Plot each class's accuracy using the colormap
for (label, accuracies), color in zip(class_accuracies.items(), colors):
    plt.plot(nodes, accuracies, label=label, linestyle='--', linewidth=3, color=color)

# Customize the plot
plt.xscale('log')  # Log scale for nodes
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(title='Classes', fontsize=30, bbox_to_anchor=(1.0, 1.0))
plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=30)
plt.yticks(fontsize=40)
# plt.xlabel("Number of Nodes", fontsize=25, fontweight='bold')
# plt.ylabel("Accuracy (%)", fontsize=25, fontweight='bold')
plt.tight_layout()

# Save the figure
plt.savefig(f"accuracy_vs_nodes_lambda_1e4-teal-orange-without-legend.png", bbox_inches='tight')
# plt.show()
plt.close()