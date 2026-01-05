import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed

# Define variables
a_values = np.array([
    -1.  , -0.98, -0.96, -0.94, -0.92, -0.9 , -0.88, -0.86, -0.84, -0.82,
    -0.8 , -0.78, -0.76, -0.74, -0.72, -0.7 , -0.68, -0.66, -0.64, -0.62,
    -0.6 , -0.58, -0.56, -0.54, -0.52, -0.5 , -0.48, -0.46, -0.44, -0.42,
    -0.4 , -0.38, -0.36, -0.34, -0.32, -0.3 , -0.28, -0.26, -0.24, -0.22,
    -0.2 , -0.18, -0.16, -0.14, -0.12, -0.1 , -0.08, -0.06, -0.04, -0.02,
     0.  ,  0.02,  0.04,  0.06,  0.08,  0.1 ,  0.12,  0.14,  0.16,  0.18,
     0.2 ,  0.22,  0.24,  0.26,  0.28,  0.3 ,  0.32,  0.34,  0.36,  0.38,
     0.4 ,  0.42,  0.44,  0.46,  0.48,  0.5 ,  0.52,  0.54,  0.56,  0.58,
     0.6 ,  0.62,  0.64,  0.66,  0.68,  0.7 ,  0.72,  0.74,  0.76,  0.78,
     0.8 ,  0.82,  0.84,  0.86,  0.88,  0.9 ,  0.92,  0.94,  0.96,  0.98,  1.0
])
u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
crossings = np.array([2.49495829, 1.24747925, 0.8316529 , 0.62374002, 0.49899218,
                      0.41582654, 0.35642278, 0.31187044, 0.2772186 , 0.24949654])
lam = 1e4
mu = 1.0
typs = ['test', 'val']
# Parameters
T = 50.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0

# calculating critical a for negative part
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
  a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

# Worker function to load data
def process_file(a, u_dc, typ, mu, lam):
    file_name = f'/scratch/almo2783/scratch/dim-less/8sensors/2600/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'
    if os.path.exists(file_name):
        try:
            data = np.loadtxt(file_name)
            return (typ, a, u_dc, data[4] * 100)  # test accuracy
        except Exception:
            return (typ, a, u_dc, np.nan)
    else:
        return (typ, a, u_dc, np.nan)

# Run in parallel
results = Parallel(n_jobs=-1, verbose=10)(
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
    
    # Normalize data for custom colormap
    data_min = np.nanmin(data_matrix)
    data_max = np.nanmax(data_matrix)
    norm = plt.Normalize(vmin=data_min, vmax=data_max)

    cmap_custom = LinearSegmentedColormap.from_list(
        "viridis_like",
        [(0, "#440154"), (0.25, "#31688e"), (0.5, "#35b779"), (0.75, "#fde725"), (1, "#fde725")]
    )
    # cmap_custom = LinearSegmentedColormap.from_list(
    #     "teal_purple",
    #     [(0, "#5ec962"), (0.5, "#21918c"), (1, "#3b528b")]
    # )
    # cmap_custom = LinearSegmentedColormap.from_list(
    #     "blue_orange",
    #     [(0, "#2166ac"), (0.5, "#f7f7f7"), (1, "#b2182b")]
    # )


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(u_dc_values, a_values, data_matrix, cmap=cmap_custom, shading='auto')
    cbar = plt.colorbar(label='Accuracy (%)')
    cbar.ax.yaxis.label.set_fontsize(20)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel('u_dc', fontsize=20)
    plt.ylabel('a', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Mark the highest value
    max_idx = np.unravel_index(np.nanargmax(data_matrix), data_matrix.shape)
    max_a = a_values[max_idx[0]]
    max_u_dc = u_dc_values[max_idx[1]]
    max_value = np.nanmax(data_matrix)

    # Compute 95th percentile threshold
    threshold_95 = np.nanpercentile(data_matrix, 99.5)

    # Find all points above threshold
    mask = data_matrix >= threshold_95
    points_a, points_u_dc = np.where(mask)

    # Plot max point
    plt.plot(
        max_u_dc, max_a, 'wo', markersize=10, markeredgewidth=2,
        markeredgecolor='black',
        label=f'Highest = {max_value:.2f} (a={max_a:.2f}, u_dc={max_u_dc:.1f})'
    )

    # Plot other top 5% points (excluding the max point to avoid duplicate marker)
    for r, c in zip(points_a, points_u_dc):
        if not (r == max_idx[0] and c == max_idx[1]):
            plt.plot(
                u_dc_values[c], a_values[r],
                'o', color='darkorange', markersize=6, alpha=0.7,
                label='Top 0.5% values' if 'Top 0.5% values' not in plt.gca().get_legend_handles_labels()[1] else ""
            )
    # Overlay a_crits curve
    plt.plot(u_dc_values, a_crits, color='red', linewidth=3)
    plt.plot(u_dc_values, crossings, color="blue", linewidth=3)
    plt.ylim(a_values.min(), a_values.max())
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'Grid-Search-{typ}.png', bbox_inches='tight', dpi=300)
    plt.close()
