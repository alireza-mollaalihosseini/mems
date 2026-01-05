import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define labels and values
class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
                "street_traffic", "tram", "bus", "metro", "park"]
lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
test_accuracies = []
class_accuracies = {label: [] for label in class_labels}  # Dictionary to store per-class accuracies

for lam in lambda_values:
    # Load test accuracy
    result = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/lambda/results/results-lambda-{lam}.txt")
    test_accuracy = result[4]
    test_accuracies.append(test_accuracy * 100)

    # Load confusion matrix and calculate class accuracies
    conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/lambda/results/conf_matrix-lambda-{lam}.txt")
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
plt.savefig(f'lambda-optimization-without-legend.png', bbox_inches='tight', dpi=900)
plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# from matplotlib.colors import LinearSegmentedColormap

# # Define labels and values
# class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
#                 "street_traffic", "tram", "bus", "metro", "park"]
# lambda_values = lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
# test_accuracies = []
# class_accuracies = {label: [] for label in class_labels}  # Dictionary to store per-class accuracies

# for lam in lambda_values:
#     # Load test accuracy
#     result = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/lambda/results/results-lambda-{lam}.txt")
#     test_accuracy = result[4]
#     test_accuracies.append(test_accuracy * 100)

#     # Load confusion matrix and calculate class accuracies
#     conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/lambda/results/conf_matrix-lambda-{lam}.txt")
#     diags = conf_matrix.diagonal()
#     class_totals = conf_matrix.sum(axis=1)
    
#     # Append class-wise accuracies to dictionary
#     for i, label in enumerate(class_labels):
#         accuracy = (diags[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
#         class_accuracies[label].append(accuracy)

# # Highest test-accuracy value
# high_test = max(test_accuracies)
# high_lambda = lambda_values[np.argmax(test_accuracies)]
# lambda_values = np.array(lambda_values)

# # Define colormap (teal → darkorange) for class accuracy lines
# cmap_custom = LinearSegmentedColormap.from_list("custom_cmap", ["teal", "darkorange"])
# num_classes = len(class_labels)
# colors = [cmap_custom(i / (num_classes - 1)) for i in range(num_classes)]  # Generate colors for each class

# # Plotting
# plt.figure(figsize=(16, 8))

# # Highlight maximum test accuracy
# plt.plot(high_lambda, high_test, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black',
#          label=f'Max. val. Acc. = {high_test:.2f}')

# # Plot test accuracies in **black**
# plt.plot(lambda_values, test_accuracies, label='Test Acc.', marker='o', color='black', linewidth=3)

# # Plot each class's accuracy using the colormap
# for (label, accuracies), color in zip(class_accuracies.items(), colors):
#     plt.plot(lambda_values, accuracies, label=f'{label}', linestyle='--', linewidth=3, color=color)

# # Customize x-axis scale and tick format
# plt.xscale('log')

# # Select 4 evenly spaced x-tick values
# xticks = np.geomspace(lambda_values.min(), lambda_values.max(), num=4)  # Select 4 log-spaced values
# plt.xticks(xticks)  # Apply the selected tick positions

# # Corrected Scientific Notation Formatting
# ax = plt.gca()  # Get the current axis
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0e}'))  # Formats as 1e4, 1e5, etc.
# ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Remove minor tick labels

# # Customize font sizes
# plt.xticks(fontsize=40)
# plt.yticks(fontsize=40)

# # Add legend
# plt.legend(fontsize=25, bbox_to_anchor=(1.0, 1.0))

# # Grid and layout adjustments
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Improves readability

# # Save the figure
# plt.tight_layout()
# plt.savefig(f'lambda-optimization.png', bbox_inches='tight', dpi=900)
# plt.close()
