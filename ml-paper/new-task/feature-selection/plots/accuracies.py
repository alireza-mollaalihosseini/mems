import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Parameters
a_value = 0.44
u_dc_value = 0.4
lambda_value = 1e-3  # Only use lambda = 1e4
top_k_values1 = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 22000]
top_k_values2 = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]

types = np.array(['abs-weights', 'random-forrest','permutation', 'removal'])

# Plotting
# plt.style.use('dark_background')
plt.figure(figsize=(16, 8))

for typ in types:

    # Initialize data storage
    test_accuracies = []

    if typ == 'abs-weights' or typ == 'random-forrest':
        top_k_values = top_k_values1
    else:
        top_k_values = top_k_values2

    # Extract accuracy data for lambda = 1e4
    for top_k in top_k_values:
        try:
            if typ == 'random-forrest':
                data = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/new-task/feature-selection/{typ}/results/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}-n_estimators-100000.txt")
            else:
                # Load test accuracy
                data = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/new-task/feature-selection/{typ}/results/results-a-{a_value:.2f}-u_dc-{u_dc_value:.2f}-topk-{top_k}-lambda-{lambda_value}.txt")
            test_accuracy = data[4] * 100  # Convert to percentage
            test_accuracies.append(test_accuracy)

        except Exception as e:
            test_accuracies.append(np.nan)
            for label in class_labels:
                class_accuracies[label].append(np.nan)  # Fill with NaNs if data is missing

    # save accuracies
    np.save(f"val-acc-topk-{typ}.npy", test_accuracies)

    # Plot test accuracy in **black**
    plt.plot(top_k_values, test_accuracies, marker='o', linewidth=2, label=f"Val. Acc. for {typ}")

# Customize the plot
plt.xscale('log')  # Log scale for nodes
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=20)
plt.xticks(top_k_values, labels=[f'{n}' for n in top_k_values], rotation=90, fontsize=15)
plt.yticks(fontsize=40)
plt.tight_layout()
plt.legend(fontsize=20)

# Save the figure
plt.savefig(f"accuracy_vs_topk.png", bbox_inches='tight')
plt.close()