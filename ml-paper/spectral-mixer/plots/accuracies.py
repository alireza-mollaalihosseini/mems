import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# a = 1.36
a = 0.44
u_dc = 0.4
# u_dc = -0.5
mu = 1.0

# Define labels and values
class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
                "street_traffic", "tram", "bus", "metro", "park"]
lambda_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10,
                     1e2, 1e3, 1e4, 1e5, 1e6])
test_accuracies = []
class_accuracies = {label: [] for label in class_labels}  # Dictionary to store per-class accuracies

for lam in lambda_values:
    # Load test accuracy
    result = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/spectral-mixer/results/results-a-{a}-u_dc-{u_dc:.2f}-lambda-{lam:.1e}.txt")
    test_accuracy = result[4]
    test_accuracies.append(test_accuracy * 100)

    # Load confusion matrix and calculate class accuracies
    conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/spectral-mixer/results/conf_matrix-a-{a}-u_dc-{u_dc:.2f}-lambda-{lam:.1e}.txt")
    diags = conf_matrix.diagonal()
    class_totals = conf_matrix.sum(axis=1)
    
    # Append class-wise accuracies to dictionary
    for i, label in enumerate(class_labels):
        accuracy = (diags[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
        class_accuracies[label].append(accuracy)

# Highest test-accuracy value
high_test = max(test_accuracies)
high_lambda = lambda_values[np.argmax(test_accuracies)]

# Define colormap (teal → darkorange) for class accuracy lines
cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ["teal", "darkorange"])
num_classes = len(class_labels)
colors = [cmap_custom(i / (num_classes - 1)) for i in range(num_classes)]  # Generate colors for each class

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(lambda_values, test_accuracies, label='Test Acc.', marker='o', color='black', linewidth=3)
plt.plot(high_lambda, high_test, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black')

# Plot each class's accuracy using the colormap
for (label, accuracies), color in zip(class_accuracies.items(), colors):
    plt.plot(lambda_values, accuracies, label=f'{label}', linestyle='--', linewidth=3, color=color)

# Customize plot
plt.xscale('log')
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
#plt.title('Test and Class-Level Accuracies vs. Lambda', fontsize=20)
# plt.legend(fontsize=30, bbox_to_anchor=(1.0, 1.0)) 
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
# plt.savefig(f'lambda-optimization-a-{a}-u_dc-{u_dc}-mu-{mu}.png', bbox_inches='tight', dpi=300)
plt.savefig(f'lambda-optimization-without-legend-a-{a}-u_dc-{u_dc}-mu-{mu}.png', bbox_inches='tight', dpi=300)
plt.close()
