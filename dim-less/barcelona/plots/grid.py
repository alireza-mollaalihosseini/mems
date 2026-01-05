import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define variables
# a_values = np.linspace(-2, 4, 101)
a_values = np.array([-2.  , -1.94, -1.88, -1.82, -1.76, -1.7 , -1.64, -1.58, -1.52,
                     -1.46, -1.4 , -1.34, -1.28, -1.22, -1.16, -1.1 , -1.04, -0.98,
                     -0.92, -0.86, -0.8 , -0.74, -0.68, -0.62, -0.56, -0.5 , -0.44,
                     -0.38, -0.32, -0.26, -0.2 , -0.14, -0.08, -0.02,  0.04,  0.1 ,
                      0.16,  0.22,  0.28,  0.34,  0.4 ,  0.46,  0.52,  0.58,  0.64,
                      0.7 ,  0.76,  0.82,  0.88,  0.94,  1.  ,  1.06,  1.12,  1.18,
                      1.24,  1.3 ,  1.36,  1.42,  1.48,  1.54,  1.6 ,  1.66,  1.72,
                      1.78,  1.84,  1.9 ,  1.96,  2.02,  2.08,  2.14,  2.2 ,  2.26,
                      2.32,  2.38,  2.44,  2.5 ,  2.56,  2.62,  2.68,  2.74,  2.8 ,
                      2.86,  2.92,  2.98,  3.04,  3.1 ,  3.16,  3.22,  3.28,  3.34,
                      3.4 ,  3.46,  3.52,  3.58,  3.64,  3.7 ,  3.76,  3.82,  3.88,
                      3.94,  4.  ])
u_dc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
missing_files = []
lam = 1e4
mu = 1.0
typs = ['test', 'val']

for typ in typs:

    # Initialize data matrix
    data_matrix = np.zeros((len(a_values), len(u_dc_values)))

    for row, a in enumerate(a_values):
        for col, u_dc in enumerate(u_dc_values):
            # Construct file name based on `a` and `u_dc`
            file_name = f'/scratch/almo2783/scratch/dim-less/barcelona/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'

            # Check if the file exists
            if os.path.exists(file_name):
                # Load the data
                # df = pd.DataFrame(np.loadtxt(file_name).T, columns=['a', 'u_dc', 'lam', 'train_accuracy', 'accuracy', 'precision', 'recall', 'f1'])
                df = pd.DataFrame(np.loadtxt(file_name))

                # Store test accuracy in the data matrix
                data_matrix[row, col] = df[0][4].mean() * 100 # df['test_accuracy'].iloc[0] * 100  # Extracting the first value
            else:
                # File does not exist, continue to the next parameters
                missing_files.append((round(a,2), float(u_dc)))

    print(missing_files)

    # Calculate the 15th percentile and set values below this to NaN (masking low percentiles)
    # percentile_15 = np.percentile(data_matrix, 77)
    # data_matrix_masked = np.where(data_matrix >= percentile_15, data_matrix, np.nan)
    data_matrix_masked = data_matrix

    # Normalize data for custom colormap
    data_min = np.nanmin(data_matrix_masked)
    data_max = np.nanmax(data_matrix_masked)

    norm = plt.Normalize(vmin=data_min, vmax=data_max)

    # Create custom colormap
    cmap_custom = LinearSegmentedColormap.from_list('custom2',
        [(norm(data_min), 'cyan'), (0.5, 'blue'), (norm(data_max), 'red')])

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the data matrix with masked values
    plt.pcolormesh(u_dc_values, a_values, data_matrix_masked, cmap=cmap_custom, shading='nearest')
    cbar = plt.colorbar(label='Accuracy (%)')
    cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.yaxis.label.set_fontsize(20)

    # Change font size and font weight of colorbar ticks for the second colorbar
    cbar.ax.tick_params(labelsize=20)  # Set fontsize and tick width
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')  # Set font weight

    # Add labels and title
    plt.xlabel('u_dc', fontweight='bold', fontsize=20)
    plt.ylabel('a', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    # plt.title('Accuracy', fontweight='bold', fontsize=20)

    # Mark the highest value on the plot and add text for a and u_dc values
    max_idx = np.unravel_index(np.nanargmax(data_matrix_masked), data_matrix_masked.shape)
    max_a = a_values[max_idx[0]]
    max_u_dc = u_dc_values[max_idx[1]]
    max_value = np.nanmax(data_matrix_masked)

    plt.plot(max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black', 
            label=f'Highest Value = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})')
    # plt.text(max_u_dc, max_a, f'({max_a:.2f}, {max_u_dc:.1f})\nMax = {max_value:.2f}', 
    #          color='black', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig(f'Grid-Search-{typ}.png', bbox_inches='tight', dpi=300)
    plt.close()

# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define variables
# a_values = np.linspace(-2, 2, 101)
# u_dc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# missing_files = []

# # Initialize data matrix
# data_matrix = np.zeros((len(a_values), len(u_dc_values)))

# for row, a in enumerate(a_values):
#     for col, u_dc in enumerate(u_dc_values):
#         # Construct file name based on `a` and `u_dc`
#         file_name = f'/scratch/almo2783/scratch/barcelona/design1/results/results-a-{a:.2f}-u_dc-{u_dc:.1f}.txt'

#         # Check if the file exists
#         if os.path.exists(file_name):
#             # Load the data
#             df = pd.DataFrame(np.loadtxt(file_name), columns=['a', 'u_dc', 'lam', 'fold', 'train_accuracy', 'test_accuracy', 'accuracy', 'precision', 'recall', 'f1'])

#             # Store test accuracy in the data matrix
#             data_matrix[row, col] = df['test_accuracy'].std() * 100 # df['test_accuracy'].iloc[0] * 100  # Extracting the first value
#         else:
#             # File does not exist, continue to the next parameters
#             missing_files.append((a, u_dc))

# print(missing_files)

# # # Calculate the 15th percentile and set values below this to NaN (masking low percentiles)
# # percentile_15 = np.percentile(data_matrix, 15)
# # data_matrix_masked = np.where(data_matrix >= percentile_15, data_matrix, np.nan)

# # Normalize data for custom colormap
# data_min = np.nanmin(data_matrix)
# data_max = np.nanmax(data_matrix)

# norm = plt.Normalize(vmin=data_min, vmax=data_max)

# # Create custom colormap
# cmap_custom = LinearSegmentedColormap.from_list('custom2',
#     [(norm(data_min), 'cyan'), (0.5, 'blue'), (norm(data_max), 'red')])

# # Plotting
# plt.figure(figsize=(10, 6))

# # Plot the data matrix with masked values
# plt.pcolormesh(u_dc_values, a_values, data_matrix, cmap=cmap_custom, shading='nearest')
# cbar = plt.colorbar()
# cbar.set_label('STD Accuracy', rotation=-90, va="bottom")
# cbar.ax.yaxis.label.set_weight('bold')
# cbar.ax.yaxis.label.set_fontsize(20)

# # Change font size and font weight of colorbar ticks for the second colorbar
# cbar.ax.tick_params(labelsize=20)  # Set fontsize and tick width
# for tick in cbar.ax.get_yticklabels():
#     tick.set_fontweight('bold')  # Set font weight

# # Add labels and title
# plt.xlabel('u_dc', fontweight='bold', fontsize=20)
# plt.ylabel('a', fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.title('STD Accuracy', fontweight='bold', fontsize=20)

# # Mark the highest value on the plot and add text for a and u_dc values
# max_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
# max_a = a_values[max_idx[0]]
# max_u_dc = u_dc_values[max_idx[1]]
# max_value = np.nanmax(data_matrix)

# # plt.plot(max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black', 
# #          label=f'Highest Value = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})')
# # plt.text(max_u_dc, max_a, f'({max_a:.2f}, {max_u_dc:.1f})\nMax = {max_value:.2f}', 
# #          color='black', ha='center', va='bottom', fontweight='bold', fontsize=10)

# # plt.legend()

# plt.tight_layout()
# plt.savefig('Grid-Search-std.png', bbox_inches='tight', dpi=300)
# plt.close()


# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define variables
# a_values = np.linspace(-2, 2, 101)
# u_dc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# missing_files = []

# # Initialize data matrices
# data_matrix = np.zeros((len(a_values), len(u_dc_values)))
# min_matrix = np.full((len(a_values), len(u_dc_values)), np.nan)  # Store minimum values

# for row, a in enumerate(a_values):
#     for col, u_dc in enumerate(u_dc_values):
#         # Construct file name based on `a` and `u_dc`
#         file_name = f'/scratch/almo2783/scratch/barcelona/design1/results/results-a-{a:.2f}-u_dc-{u_dc:.1f}.txt'

#         # Check if the file exists
#         if os.path.exists(file_name):
#             # Load the data
#             df = pd.DataFrame(np.loadtxt(file_name), columns=['a', 'u_dc', 'lam', 'fold', 'train_accuracy', 'test_accuracy', 'accuracy', 'precision', 'recall', 'f1'])

#             # Store mean test accuracy in the data matrix
#             data_matrix[row, col] = df['test_accuracy'].mean() * 100
            
#             # Store the minimum test accuracy among folds
#             min_matrix[row, col] = df['test_accuracy'].min() * 100
#         else:
#             missing_files.append((round(a, 2), float(u_dc)))

# print(missing_files)

# # Normalize data for custom colormap
# data_min = np.nanmin(data_matrix)
# data_max = np.nanmax(data_matrix)

# norm = plt.Normalize(vmin=data_min, vmax=data_max)

# # Create custom colormap
# cmap_custom = LinearSegmentedColormap.from_list('custom2',
#     [(norm(data_min), 'cyan'), (0.5, 'blue'), (norm(data_max), 'red')])

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(u_dc_values, a_values, data_matrix, cmap=cmap_custom, shading='nearest')
# cbar = plt.colorbar(label='Accuracy (%)')
# cbar.ax.yaxis.label.set_weight('bold')
# cbar.ax.yaxis.label.set_fontsize(20)

# # Customize colorbar ticks
# cbar.ax.tick_params(labelsize=20)
# for tick in cbar.ax.get_yticklabels():
#     tick.set_fontweight('bold')

# # Add labels and title
# plt.xlabel('u_dc', fontweight='bold', fontsize=20)
# plt.ylabel('a', fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.title('Validation Accuracy', fontweight='bold', fontsize=20)

# # Mark the highest mean accuracy on the plot
# max_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
# max_a = a_values[max_idx[0]]
# max_u_dc = u_dc_values[max_idx[1]]
# max_value = np.nanmax(data_matrix)
# plt.plot(max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black',
#          label=f'Highest Value = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})')

# # Mark the highest minimum test accuracy on the plot
# min_max_idx = np.unravel_index(np.nanargmax(min_matrix), min_matrix.shape)
# min_max_a = a_values[min_max_idx[0]]
# min_max_u_dc = u_dc_values[min_max_idx[1]]
# min_max_value = np.nanmax(min_matrix)
# plt.plot(min_max_u_dc, min_max_a, 'go', markersize=10, markeredgewidth=2, markeredgecolor='black',
#          label=f'Highest Min Value = {min_max_value:.2f} (a={min_max_a:.2f}, u_dc={min_max_u_dc:.1f})')

# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig('Grid-Search1.png', bbox_inches='tight', dpi=300)
# plt.close()
