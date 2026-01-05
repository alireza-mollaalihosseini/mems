import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# a = 1.36
# a = 0.44
u_dc = 0.4
# u_dc = -0.5
mu = 1.0
lam = 1e5
a_values = np.linspace(-1, 1, 101)

# Define labels and values
class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
                "street_traffic", "tram", "bus", "metro", "park"]

test_accuracies = []
class_accuracies = {label: [] for label in class_labels}  # Dictionary to store per-class accuracies

for a in a_values:
    # Load test accuracy
    result = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/time-domain/fixed-u_dc/results/results_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt")
    test_accuracy = result[4]
    test_accuracies.append(test_accuracy * 100)

    # Load confusion matrix and calculate class accuracies
    conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/ml-paper/time-domain/fixed-u_dc/results/conf_matrix_val-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt")
    diags = conf_matrix.diagonal()
    class_totals = conf_matrix.sum(axis=1)
    
    # Append class-wise accuracies to dictionary
    for i, label in enumerate(class_labels):
        accuracy = (diags[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
        class_accuracies[label].append(accuracy)

# save validation results
np.save('val_acc.npy', test_accuracies)

# Highest test-accuracy value
high_test = max(test_accuracies)
high_a = a_values[np.argmax(test_accuracies)]

# Define colormap (teal → darkorange) for class accuracy lines
cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ["teal", "darkorange"])
num_classes = len(class_labels)
colors = [cmap_custom(i / (num_classes - 1)) for i in range(num_classes)]  # Generate colors for each class

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(a_values, test_accuracies, label='Val Acc.', marker='o', color='black', linewidth=3)
plt.plot(high_a, high_test, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black')

# Plot each class's accuracy using the colormap
for (label, accuracies), color in zip(class_accuracies.items(), colors):
    plt.plot(a_values, accuracies, label=f'{label}', linestyle='--', linewidth=3, color=color)

# Customize plot
# plt.xscale('log')
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
#plt.title('Test and Class-Level Accuracies vs. Lambda', fontsize=20)
# plt.legend(fontsize=30, bbox_to_anchor=(1.0, 1.0)) 
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
# plt.savefig(f'lambda-optimization-a-{a}-u_dc-{u_dc}-mu-{mu}.png', bbox_inches='tight', dpi=300)
plt.savefig(f'accuracies-a-{a}-u_dc-{u_dc}-mu-{mu}.png', bbox_inches='tight', dpi=300)
plt.close()
