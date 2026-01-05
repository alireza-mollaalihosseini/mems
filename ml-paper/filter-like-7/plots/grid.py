# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define variables
# a_values = np.array([-1.  , -0.98, -0.96, -0.94, -0.92, -0.9 , -0.88, -0.86, -0.84,
#                      -0.82, -0.8 , -0.78, -0.76, -0.74, -0.72, -0.7 , -0.68, -0.66,
#                      -0.64, -0.62, -0.6 , -0.58, -0.56, -0.54, -0.52, -0.5 , -0.48,
#                      -0.46, -0.44, -0.42, -0.4 , -0.38, -0.36, -0.34, -0.32, -0.3 ,
#                      -0.28, -0.26, -0.24, -0.22, -0.2 , -0.18, -0.16, -0.14, -0.12,
#                      -0.1 , -0.08, -0.06, -0.04, -0.02,  0.  ,  0.02,  0.04,  0.06,
#                       0.08,  0.1 ,  0.12,  0.14,  0.16,  0.18,  0.2 ,  0.22,  0.24,
#                       0.26,  0.28,  0.3 ,  0.32,  0.34,  0.36,  0.38,  0.4 ,  0.42,
#                       0.44,  0.46,  0.48,  0.5 ,  0.52,  0.54,  0.56,  0.58,  0.6 ,
#                       0.62,  0.64,  0.66,  0.68,  0.7 ,  0.72,  0.74,  0.76,  0.78,
#                       0.8 ,  0.82,  0.84,  0.86,  0.88,  0.9 ,  0.92,  0.94,  0.96,
#                       0.98,  1.  ])
# u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# missing_files = []
# lam = 1e4
# mu = 1.0
# typs = ['test', 'val']

# for typ in typs:

#     # Initialize data matrix
#     data_matrix = np.zeros((len(a_values), len(u_dc_values)))

#     for row, a in enumerate(a_values):
#         for col, u_dc in enumerate(u_dc_values):
#             # Construct file name based on `a` and `u_dc`
#             file_name = f'/scratch/almo2783/scratch/dim-less/8sensors/445/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'

#             # Check if the file exists
#             if os.path.exists(file_name):
#                 # Load the data
#                 # df = pd.DataFrame(np.loadtxt(file_name).T, columns=['a', 'u_dc', 'lam', 'train_accuracy', 'accuracy', 'precision', 'recall', 'f1'])
#                 df = pd.DataFrame(np.loadtxt(file_name))

#                 # Store test accuracy in the data matrix
#                 data_matrix[row, col] = df[0][4].mean() * 100 # df['test_accuracy'].iloc[0] * 100  # Extracting the first value
#             else:
#                 # File does not exist, continue to the next parameters
#                 missing_files.append((round(a,2), float(u_dc)))

#     print(missing_files)

#     # Calculate the 15th percentile and set values below this to NaN (masking low percentiles)
#     # percentile_15 = np.percentile(data_matrix, 77)
#     # data_matrix_masked = np.where(data_matrix >= percentile_15, data_matrix, np.nan)
#     data_matrix_masked = data_matrix

#     # Normalize data for custom colormap
#     data_min = np.nanmin(data_matrix_masked)
#     data_max = np.nanmax(data_matrix_masked)

#     norm = plt.Normalize(vmin=data_min, vmax=data_max)

#     # Create custom colormap
#     cmap_custom = LinearSegmentedColormap.from_list('custom2',
#         [(norm(data_min), 'cyan'), (0.5, 'blue'), (norm(data_max), 'red')])

#     # Plotting
#     plt.figure(figsize=(10, 6))

#     # Plot the data matrix with masked values
#     plt.pcolormesh(u_dc_values, a_values, data_matrix_masked, cmap=cmap_custom, shading='nearest')
#     cbar = plt.colorbar(label='Accuracy (%)')
#     cbar.ax.yaxis.label.set_weight('bold')
#     cbar.ax.yaxis.label.set_fontsize(20)

#     # Change font size and font weight of colorbar ticks for the second colorbar
#     cbar.ax.tick_params(labelsize=20)  # Set fontsize and tick width
#     for tick in cbar.ax.get_yticklabels():
#         tick.set_fontweight('bold')  # Set font weight

#     # Add labels and title
#     plt.xlabel('u_dc', fontweight='bold', fontsize=20)
#     plt.ylabel('a', fontweight='bold', fontsize=20)
#     plt.xticks(fontweight='bold', fontsize=20)
#     plt.yticks(fontweight='bold', fontsize=20)
#     # plt.title('Accuracy', fontweight='bold', fontsize=20)

#     # Mark the highest value on the plot and add text for a and u_dc values
#     max_idx = np.unravel_index(np.nanargmax(data_matrix_masked), data_matrix_masked.shape)
#     max_a = a_values[max_idx[0]]
#     max_u_dc = u_dc_values[max_idx[1]]
#     max_value = np.nanmax(data_matrix_masked)

#     plt.plot(max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black', 
#             label=f'Highest Value = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})')
#     # plt.text(max_u_dc, max_a, f'({max_a:.2f}, {max_u_dc:.1f})\nMax = {max_value:.2f}', 
#     #          color='black', ha='center', va='bottom', fontweight='bold', fontsize=10)

#     plt.legend(fontsize=15)

#     plt.tight_layout()
#     plt.savefig(f'Grid-Search-{typ}.png', bbox_inches='tight', dpi=300)
#     plt.close()

# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define variables
# a_values = np.array([-1.  , -0.98, -0.96, -0.94, -0.92, -0.9 , -0.88, -0.86, -0.84,
#                      -0.82, -0.8 , -0.78, -0.76, -0.74, -0.72, -0.7 , -0.68, -0.66,
#                      -0.64, -0.62, -0.6 , -0.58, -0.56, -0.54, -0.52, -0.5 , -0.48,
#                      -0.46, -0.44, -0.42, -0.4 , -0.38, -0.36, -0.34, -0.32, -0.3 ,
#                      -0.28, -0.26, -0.24, -0.22, -0.2 , -0.18, -0.16, -0.14, -0.12,
#                      -0.1 , -0.08, -0.06, -0.04, -0.02,  0.  ,  0.02,  0.04,  0.06,
#                       0.08,  0.1 ,  0.12,  0.14,  0.16,  0.18,  0.2 ,  0.22,  0.24,
#                       0.26,  0.28,  0.3 ,  0.32,  0.34,  0.36,  0.38,  0.4 ,  0.42,
#                       0.44,  0.46,  0.48,  0.5 ,  0.52,  0.54,  0.56,  0.58,  0.6 ,
#                       0.62,  0.64,  0.66,  0.68,  0.7 ,  0.72,  0.74,  0.76,  0.78,
#                       0.8 ,  0.82,  0.84,  0.86,  0.88,  0.9 ,  0.92,  0.94,  0.96,
#                       0.98,  1.  ])
# u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# missing_files = []
# lam = 1e4
# mu = 1.0
# typs = ['test', 'val']

# for typ in typs:

#     # Initialize data matrix
#     data_matrix = np.zeros((len(a_values), len(u_dc_values)))

#     for row, a in enumerate(a_values):
#         for col, u_dc in enumerate(u_dc_values):
#             # Construct file name based on `a` and `u_dc`
#             file_name = f'/scratch/almo2783/scratch/dim-less/8sensors/445/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'

#             # Check if the file exists
#             if os.path.exists(file_name):
#                 data = np.loadtxt(file_name)

#                 # Store test accuracy in the data matrix
#                 data_matrix[row, col] = data[4] * 100 # df['test_accuracy'].iloc[0] * 100  # Extracting the first value
#             else:
#                 # File does not exist, continue to the next parameters
#                 missing_files.append((round(a,2), float(u_dc)))

#     print(missing_files)

#     data_matrix_masked = data_matrix

#     # Normalize data for custom colormap
#     data_min = np.nanmin(data_matrix_masked)
#     data_max = np.nanmax(data_matrix_masked)

#     norm = plt.Normalize(vmin=data_min, vmax=data_max)

#     cmap_custom = LinearSegmentedColormap.from_list(
#         "viridis_like",
#         [(0, "#440154"), (0.25, "#31688e"), (0.5, "#35b779"), (0.75, "#fde725"), (1, "#fde725")]
#     )

#     # Plotting
#     plt.figure(figsize=(10, 6))

#     # Plot the data matrix with masked values
#     plt.pcolormesh(u_dc_values, a_values, data_matrix_masked, cmap=cmap_custom, shading='nearest')
#     cbar = plt.colorbar(label='Accuracy (%)')
#     cbar.ax.yaxis.label.set_fontsize(20)

#     # Change font size and font weight of colorbar ticks for the second colorbar
#     cbar.ax.tick_params(labelsize=20)  # Set fontsize and tick width

#     # Add labels and title
#     plt.xlabel('u_dc', fontsize=20)
#     plt.ylabel('a', fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)

#     # Mark the highest value on the plot and add text for a and u_dc values
#     max_idx = np.unravel_index(np.nanargmax(data_matrix_masked), data_matrix_masked.shape)
#     max_a = a_values[max_idx[0]]
#     max_u_dc = u_dc_values[max_idx[1]]
#     max_value = np.nanmax(data_matrix_masked)

#     plt.plot(max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2, markeredgecolor='black', 
#             label=f'Highest Value = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})')

#     plt.legend(fontsize=15)

#     plt.tight_layout()
#     plt.savefig(f'Grid-Search-{typ}.png', bbox_inches='tight', dpi=300)
#     plt.close()


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed

# Define variables
a_values = np.array([-2.  , -1.96, -1.92, -1.88, -1.84, -1.8 , -1.76, -1.72, -1.68,
                     -1.64, -1.6 , -1.56, -1.52, -1.48, -1.44, -1.4 , -1.36, -1.32,
                     -1.28, -1.24, -1.2 , -1.16, -1.12, -1.08, -1.04, -1.  , -0.96,
                     -0.92, -0.88, -0.84, -0.8 , -0.76, -0.72, -0.68, -0.64, -0.6 ,
                     -0.56, -0.52, -0.48, -0.44, -0.4 , -0.36, -0.32, -0.28,
                     -0.24, -0.2 , -0.16, -0.12, -0.08, -0.04,  0.  ,  0.04,  0.08,
                      0.12,  0.16,  0.2 ,  0.24,  0.28,  0.32,  0.36,  0.4 ,
                      0.44,  0.48,  0.52,  0.56,  0.6 ,  0.64,  0.68,  0.72,  0.76,
                      0.8 ,  0.84,  0.88,  0.92,  0.96,  1.  ,  1.04,  1.08,
                      1.12,  1.16,  1.2 ,  1.24,  1.28,  1.32,  1.36,  1.4 ,  1.44,
                      1.48,  1.52,  1.56,  1.6 ,  1.64,  1.68,  1.72,  1.76,  1.8 ,
                      1.84,  1.88,  1.92,  1.96,  2.  ])
# a_values = np.linspace(-2,2,101)
u_dc_values = np.array([0.4])
# crossings = np.array([0.97950606, 0.4897876 , 0.32663367, 0.24493265, 0.19610222,
#                       0.16344693, 0.14025635, 0.12251293, 0.10901716, 0.09833336])
lam = 1e4
mu = 1.0
typs = ['test', 'val']

# Parameters
T = 50.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 445.0, 2796.0174616949157, 29.0, 0.0034, 294.11764705882354, 2256560.0, 13.4, 147800.0

# # calculating critical a for negative part
# a_crits = np.zeros(len(u_dc_values))
# for i, u_dc in enumerate(u_dc_values):
#   a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )


# Worker function to load data
def process_file(a, u_dc, typ, mu, lam):
    file_name = f'/scratch/almo2783/scratch/ml-paper/filter-like-7/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'
    if os.path.exists(file_name):
        try:
            data = np.loadtxt(file_name)
            return (typ, a, u_dc, data[4] * 100)  # test accuracy
        except Exception:
            return (typ, a, u_dc, np.nan)
    else:
        return (typ, a, u_dc, np.nan)

# Run in parallel
results = Parallel(n_jobs=-1, backend='threading', verbose=10)(
    delayed(process_file)(a, u_dc, typ, mu, lam)
    for typ in typs
    for a in a_values
    for u_dc in u_dc_values
)

# Process results for each type
for typ in typs:
    data_matrix = np.full((len(a_values), len(u_dc_values)), np.nan)
    missing_files = []

    # Filter results for this typ
    filtered = [r for r in results if r[0] == typ]

    for _, a, u_dc, acc in filtered:
        row = np.where(a_values == a)[0][0]
        col = np.where(u_dc_values == u_dc)[0][0]
        if np.isnan(acc):
            missing_files.append((round(a, 2), float(u_dc)))
        data_matrix[row, col] = acc

    print(f"Missing files for {typ}: {missing_files}")

    # save the data_matrix
    np.save(f"data_matrix_{typ}.npy", data_matrix)

    plt.figure(figsize=(16,8))
    plt.plot(a_values, data_matrix)
    plt.scatter(a_values, data_matrix)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'accuracy-{typ}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # # Normalize data for custom colormap
    # data_min = np.nanmin(data_matrix)
    # data_max = np.nanmax(data_matrix)
    # norm = plt.Normalize(vmin=data_min, vmax=data_max)

    # cmap_custom = LinearSegmentedColormap.from_list(
    #     "viridis_like",
    #     [(0, "#440154"), (0.25, "#31688e"), (0.5, "#35b779"), (0.75, "#fde725"), (1, "#fde725")]
    # )
    # # cmap_custom = LinearSegmentedColormap.from_list(
    # #     "teal_purple",
    # #     [(0, "#5ec962"), (0.5, "#21918c"), (1, "#3b528b")]
    # # )
    # # cmap_custom = LinearSegmentedColormap.from_list(
    # #     "blue_orange",
    # #     [(0, "#2166ac"), (0.5, "#f7f7f7"), (1, "#b2182b")]
    # # )


    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(u_dc_values, a_values, data_matrix, cmap=cmap_custom, shading='auto')
    # cbar = plt.colorbar(label='Accuracy (%)')
    # cbar.ax.yaxis.label.set_fontsize(20)
    # cbar.ax.tick_params(labelsize=20)

    # plt.xlabel('u_dc', fontsize=20)
    # plt.ylabel('a', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    # # Mark the highest value
    # max_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
    # max_a = a_values[max_idx[0]]
    # max_u_dc = u_dc_values[max_idx[1]]
    # max_value = np.nanmax(data_matrix)

    # # Compute 95th percentile threshold
    # threshold_95 = np.nanpercentile(data_matrix, 99.5)

    # # Find all points above threshold
    # mask = data_matrix >= threshold_95
    # points_a, points_u_dc = np.where(mask)

    # # Plot max point
    # plt.plot(
    #     max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2,
    #     markeredgecolor='black',
    #     label=f'Highest = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})'
    # )

    # # Plot other top 5% points (excluding the max point to avoid duplicate marker)
    # for r, c in zip(points_a, points_u_dc):
    #     if not (r == max_idx[0] and c == max_idx[1]):
    #         plt.plot(
    #             u_dc_values[c], a_values[r],
    #             'o', color='darkorange', markersize=6, alpha=0.7,
    #             label='Top 0.5% values' if 'Top 0.5% values' not in plt.gca().get_legend_handles_labels()[1] else ""
    #         )

    # # Overlay a_crits curve
    # # plt.plot(u_dc_values, a_crits, color='red', linewidth=3)
    # # plt.plot(u_dc_values, crossings, color="blue", linewidth=3)
    # plt.ylim(a_values.min(), a_values.max())
    # plt.legend(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'Grid-Search-{typ}.png', bbox_inches='tight', dpi=300)
    # plt.close()
