import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Parameters
a_value = 0.44
u_dc_value = 0.4
lambda_value = 1e4  # Only use lambda = 1e4
top_k_values = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]

# Length of extracted features data
feature_lengths = {
    1e-6: {10: 10, 20: 20, 50: 50, 100: 100, 200: 200, 300: 300, 400: 400, 500: 500,
           600: 600, 700: 700, 800: 800, 900: 900, 1000: 1000, 1500: 1500, 2000: 2000,
           3000: 3000, 4000: 4000, 5000: 5000, 10000: 10000, 20000: 20000, 22000: 22000}
}

types = np.array(['abs-weights','abs', 'norm', 'random-forrest', 'anova-test', 'mutual-info', 'lasso', 'mlp', 'rfe', 'xgboost'])

# Plotting
# plt.style.use('dark_background')
plt.figure(figsize=(16, 8))

# Extract node values for x-axis
nodes = [feature_lengths[1e-6].get(top_k, 0) for top_k in top_k_values]

for typ in types:

    # Initialize data storage
    test_accuracies = []

    # Extract accuracy data for lambda = 1e4
    for top_k in top_k_values:
        try:
            # Load test accuracy
            data = np.loadtxt(f"/scratch/almo2783/scratch/dim-less/8sensors/feature-par/{typ}/2600/results/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt")
            test_accuracy = data[4] * 100  # Convert to percentage
            test_accuracies.append(test_accuracy)

        except Exception as e:
            test_accuracies.append(np.nan)
            for label in class_labels:
                class_accuracies[label].append(np.nan)  # Fill with NaNs if data is missing

    # Plot test accuracy in **black**
    plt.plot(nodes, test_accuracies, marker='o', linewidth=2, label=f"Val. Acc. for {typ}")

# Customize the plot
plt.xscale('log')  # Log scale for nodes
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=20)
plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=15)
plt.yticks(fontsize=40)
plt.tight_layout()

# Save the figure
plt.savefig(f"accuracy_vs_nodes_lambda_1e4-new-without-legend.png", bbox_inches='tight')
plt.close()