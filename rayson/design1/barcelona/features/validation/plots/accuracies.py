# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# a_value = -0.56
# steps = [1e-4, 1e-5, 1.0 / 44100.0, 1e-6]
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
# lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# # Iterate over each step
# for step in steps:
#     accuracies = []
    
#     # Collect accuracies for each alpha and lambda pair
#     for alpha in alpha_values:
#         row_accuracies = []
#         for lambda_value in lambda_values:
#             try:
#                 # Load the data
#                 data = np.loadtxt(f"/scratch/almo2783/scratch/features/design1/results/results-step-{step}-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#                 test_accuracy = data[4] * 100
#                 row_accuracies.append(test_accuracy)
#             except Exception as e:
#                 row_accuracies.append(np.nan)  # Handle missing data gracefully
#         accuracies.append(row_accuracies)
    
#     # Convert to a numpy array
#     accuracies = np.array(accuracies)
    
#     # Plot the heatmap
#     plt.figure(figsize=(12, 8))
#     plt.imshow(accuracies, aspect='auto', cmap='viridis', extent=[min(lambda_values), max(lambda_values), max(alpha_values), min(alpha_values)])
#     plt.colorbar(label='Test Accuracy (%)')
#     plt.xlabel('Lambda Values', fontsize=14)
#     plt.ylabel('Alpha Values', fontsize=14)
#     plt.title(f'Accuracy Heatmap for Step {step:.1e}', fontsize=16)
#     plt.xscale('log')
#     plt.xticks(lambda_values, rotation=45, fontsize=10)
#     plt.yticks(alpha_values, fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f"heatmap_step_{step:.1e}.png", dpi=300)
#     # plt.show()
#     plt.close()



# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# a_value = -0.56
# steps = [1e-4, 1e-5, 1.0 / 44100.0, 1e-6]
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
# lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# # Iterate over each step
# for step in steps:
#     plt.figure(figsize=(12, 8))

#     # Plot trends for each alpha value
#     for alpha in alpha_values:
#         accuracies = []
#         for lambda_value in lambda_values:
#             try:
#                 # Load the data
#                 data = np.loadtxt(f"/scratch/almo2783/scratch/features/design1/results/results-step-{step}-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#                 test_accuracy = data[4] * 100
#                 accuracies.append(test_accuracy)
#             except Exception as e:
#                 accuracies.append(np.nan)  # Handle missing data gracefully

#         # Plot the line for the current alpha
#         plt.plot(lambda_values, accuracies, label=f'Alpha = {alpha}', marker='o', linewidth=2)

#     # Customize the plot
#     plt.xscale('log')
#     plt.xlabel('Lambda Values', fontsize=14, fontweight='bold')
#     plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
#     plt.title(f'Trends for Step {step:.1e}', fontsize=16, fontweight='bold')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(title='Alpha Values', fontsize=10)
#     plt.xticks(lambda_values, labels=[f'{v:.0e}' for v in lambda_values], rotation=45, fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.tight_layout()

#     # Save and show the plot
#     plt.savefig(f"trends_step_{step:.1e}.png", dpi=300)
#     # plt.show()
#     plt.close()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# # Parameters
# a_value = -0.56
# steps = [1e-4, 1e-5, 1.0 / 44100.0, 1e-6]
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
# lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# # Iterate over each step
# for step in steps:
#     plt.figure(figsize=(12, 8))

#     # Create a colormap for the alpha values
#     colormap = cm.get_cmap('tab20', len(alpha_values))  # 'tab20' can handle up to 20 unique colors

#     # Plot trends for each alpha value
#     for idx, alpha in enumerate(alpha_values):
#         accuracies = []
#         for lambda_value in lambda_values:
#             try:
#                 # Load the data
#                 data = np.loadtxt(f"/scratch/almo2783/scratch/features/design1/results/results-step-{step}-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#                 test_accuracy = data[4] * 100
#                 accuracies.append(test_accuracy)
#             except Exception as e:
#                 accuracies.append(np.nan)

#         # Skip plotting if all accuracies are NaN
#         if not any(np.isfinite(accuracies)):
#             continue

#         # Use colormap to assign colors dynamically
#         color = colormap(idx / len(alpha_values))
#         plt.plot(lambda_values, accuracies, label=f'Alpha = {alpha}', marker='o', linewidth=2, color=color)

#     # Customize the plot
#     plt.xscale('log')
#     plt.xlabel('Lambda Values', fontsize=14, fontweight='bold')
#     plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
#     plt.title(f'Trends for Step {step:.1e}', fontsize=16, fontweight='bold')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(title='Alpha Values', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.xticks(lambda_values, labels=[f'{v:.0e}' for v in lambda_values], rotation=45, fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.tight_layout()

#     # Save and show the plot
#     plt.savefig(f"trends_step_{step:.1e}.png", dpi=300, bbox_inches='tight')
#     # plt.show()
#     plt.close()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# # Parameters
# a_value = -1.08
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
# lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# # Length of extracted features data
# feature_lengths = {
#     1e-6: {1: 23705, 2: 18421, 3: 9503, 4: 4117, 5: 2073, 6: 1260, 7: 905, 8: 686, 9: 550, 10: 467, 15: 242, 20: 146, 50: 22}
# }


# plt.figure(figsize=(16, 8))

# # Create a colormap for the alpha values
# colormap = cm.get_cmap('tab20', len(alpha_values))

# # Plot trends for each alpha value
# for idx, alpha in enumerate(alpha_values):
#     accuracies = []
#     for lambda_value in lambda_values:
#         try:
#             # Load the data
#             data = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/validation/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#             test_accuracy = data[4] * 100
#             accuracies.append(test_accuracy)
#         except Exception as e:
#             accuracies.append(np.nan)

#     # Skip plotting if all accuracies are NaN
#     if not any(np.isfinite(accuracies)):
#         continue

#     # Use colormap to assign colors dynamically
#     color = colormap(idx / len(alpha_values))
    
#     # Update legend with nodes instead of alpha
#     nodes = feature_lengths[1e-6].get(alpha, 0)
#     plt.plot(lambda_values, accuracies, label=f'Nodes = {nodes}', marker='o', linewidth=2, color=color)

# # Customize the plot
# plt.xscale('log')
# plt.xlabel('Lambda Values', fontsize=14, fontweight='bold')
# plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
# plt.title(f'Trends', fontsize=16, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title=f'Nodes', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(lambda_values, labels=[f'{v:.0e}' for v in lambda_values], rotation=45, fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()

# # Save and show the plot
# plt.savefig(f"trends.png", dpi=300, bbox_inches='tight')
# # plt.show()
# plt.close()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Parameters
a_value = -1.08
lambda_value = 1e4  # Only use lambda = 1e4
alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
                "street_traffic", "tram", "bus", "metro", "park"]

# Length of extracted features data
feature_lengths = {
    1e-6: {1: 23705, 2: 18421, 3: 9503, 4: 4117, 5: 2073, 6: 1260, 7: 905, 8: 686, 9: 550, 10: 467, 15: 242, 20: 146, 50: 22}
}

# Initialize data storage
test_accuracies = []
class_accuracies = {label: [] for label in class_labels}  # Store per-class accuracies

# Extract accuracy data for lambda = 1e4
for alpha in alpha_values:
    try:
        # Load test accuracy
        data = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/validation/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
        test_accuracy = data[4] * 100  # Convert to percentage
        test_accuracies.append(test_accuracy)

        # Load confusion matrix and extract per-class accuracies
        conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/validation/results/conf_matrix-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
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
plt.savefig(f"accuracy_vs_nodes_lambda_1e4-teal-orange-without-legend.png", dpi=900, bbox_inches='tight')
# plt.show()
plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# # Parameters
# a_value = -1.08
# lambda_value = 1e3  # Only use lambda = 1e4
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
# class_labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", 
#                 "street_traffic", "tram", "bus", "metro", "park"]

# # Length of extracted features data
# feature_lengths = {
#     1e-6: {1: 23705, 2: 18421, 3: 9503, 4: 4117, 5: 2073, 6: 1260, 7: 905, 8: 686, 9: 550, 10: 467, 15: 242, 20: 146, 50: 22}
# }

# # Initialize data storage
# test_accuracies = []
# class_accuracies = {label: [] for label in class_labels}  # Store per-class accuracies

# # Extract accuracy data for lambda = 1e4
# for alpha in alpha_values:
#     try:
#         # Load test accuracy
#         data = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/validation/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#         test_accuracy = data[4] * 100  # Convert to percentage
#         test_accuracies.append(test_accuracy)

#         # Load confusion matrix and extract per-class accuracies
#         conf_matrix = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/barcelona/features/validation/results/conf_matrix-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
#         diags = conf_matrix.diagonal()
#         class_totals = conf_matrix.sum(axis=1)

#         for i, label in enumerate(class_labels):
#             accuracy = (diags[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
#             class_accuracies[label].append(accuracy)

#     except Exception as e:
#         test_accuracies.append(np.nan)
#         for label in class_labels:
#             class_accuracies[label].append(np.nan)  # Fill with NaNs if data is missing

# # Extract node values for x-axis
# nodes = [feature_lengths[1e-6].get(alpha, 0) for alpha in alpha_values]

# # Plotting
# plt.figure(figsize=(16, 8))

# # Plot test accuracy in **black**
# plt.plot(nodes, test_accuracies, marker='o', linewidth=2, color='black', label="Val. Acc.")

# # Define colormap for class accuracy lines
# cmap_custom = cm.get_cmap('tab20', len(class_labels))  # Use tab20 for more colors

# # Plot each class's accuracy using the colormap
# for idx, (label, accuracies) in enumerate(class_accuracies.items()):
#     plt.plot(nodes, accuracies, label=label, linestyle='--', linewidth=3, color=cmap_custom(idx / len(class_labels)))

# # Customize the plot
# plt.xscale('log')  # Log scale for nodes
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xticks(nodes, labels=[f'{n}' for n in nodes], rotation=90, fontsize=20)
# plt.yticks(fontsize=30)
# plt.legend(fontsize=30, bbox_to_anchor=(1.0, 1), loc='upper left')
# # plt.xlabel("Number of Nodes", fontsize=30, fontweight="bold")
# # plt.ylabel("Accuracy (%)", fontsize=30, fontweight="bold")
# plt.tight_layout()

# # Save the figure
# plt.savefig("accuracy_vs_nodes_lambda_1e3-1-.png", dpi=900, bbox_inches='tight')
# plt.close()