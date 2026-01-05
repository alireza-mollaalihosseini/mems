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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
a_value = -1.08
alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# Length of extracted features data
feature_lengths = {
    1e-6: {1: 23795, 2: 18977, 3: 9787, 4: 3810, 5: 1712, 6: 995, 7: 680, 8: 509, 9: 419, 10: 334, 15: 163, 20: 84, 50: 9}
}


plt.figure(figsize=(12, 8))

# Create a colormap for the alpha values
colormap = cm.get_cmap('tab20', len(alpha_values))

# Plot trends for each alpha value
for idx, alpha in enumerate(alpha_values):
    accuracies = []
    for lambda_value in lambda_values:
        try:
            # Load the data
            data = np.loadtxt(f"/scratch/almo2783/scratch/rayson/design1/3cities/features/results/results-a-{a_value:.2f}-alpha-{alpha}-lambda-{lambda_value}.txt")
            test_accuracy = data[4] * 100
            accuracies.append(test_accuracy)
        except Exception as e:
            accuracies.append(np.nan)

    # Skip plotting if all accuracies are NaN
    if not any(np.isfinite(accuracies)):
        continue

    # Use colormap to assign colors dynamically
    color = colormap(idx / len(alpha_values))
    
    # Update legend with nodes instead of alpha
    nodes = feature_lengths[1e-6].get(alpha, 0)
    plt.plot(lambda_values, accuracies, label=f'Nodes = {nodes}', marker='o', linewidth=2, color=color)

# Customize the plot
plt.xscale('log')
plt.xlabel('Lambda Values', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title(f'Trends', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title=f'Nodes', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(lambda_values, labels=[f'{v:.0e}' for v in lambda_values], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Save and show the plot
plt.savefig(f"trends.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()
