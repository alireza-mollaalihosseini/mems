# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# import matplotlib.colors as mcolors

# # Parameters
# T = 500.0                      # Total simulation time (seconds)
# t_rec = 3.0                    # Recording time (seconds)
# noise = 2e-4                  # Noise magnitude
# u_max = 1.0

# omega_0 = 53956.46373431294
# Q_0     = 50.0
# alpha   = 19.2
# beta    = 1066.0
# gamma   = 1.62e7
# R       = 12.5
# tau     = 0.001
# kappa   = 0.602e6

# a_values = np.linspace(-3, 4, 1001)
# a_values_flipped = a_values[::-1]

# u_dc_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.11,
#                         0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21, 0.22,
#                         0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33,
#                         0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44,
#                         0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55,
#                         0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66,
#                         0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
#                         0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88,
#                         0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])

# # Threshold for treating time series as constant
# const_tol = 1e-8

# # Storage array for number of unique extrema
# extrema_counts = np.zeros((len(a_values), len(u_dc_values)), dtype=int)

# for j, u_dc in enumerate(u_dc_values):
#     # Load precomputed results for this u_dc
#     results_down = np.load(
#         f'/scratch/almo2783/scratch/dim-less/grid-search/deflections-down/'
#         f'deflections-RK4-down-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-04-u_max-{u_max}-u_dc-{u_dc}.npy'
#     )

#     for i, a in enumerate(a_values_flipped):
#         data = results_down[i, :]
#         if np.std(data) < const_tol:
#             extrema_counts[i, j] = 0
#             # print((a, data_up[-1]))
#             continue

#         data = data[-50000:]
#         peaks, _ = find_peaks(data)
#         maxima = data[peaks]
#         unique_maxima = np.unique(maxima.round(decimals=4))
#         if np.std(unique_maxima) < 0.01:
#             extrema_counts[i, j] = 1
#             continue

#         extrema_counts[i, j] = len(unique_maxima)


# # calculating critical a for negative part
# a_crits = np.zeros(len(u_dc_values))
# for i, u_dc in enumerate(u_dc_values):
#   a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

# # Mask out zeros (keep them black)
# masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

# # Get unique values excluding 0
# unique_vals = np.unique(extrema_counts[extrema_counts > 0])

# # Build a discrete colormap (start from viridis but only pick needed number of colors)
# colors = plt.cm.viridis(np.linspace(0, 1, len(unique_vals)))
# cmap = mcolors.ListedColormap(colors)
# cmap.set_bad(color="black")  # for masked values (zeros)

# # Normalize so each integer falls into its own color bin
# bounds = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5, 1)
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# # --- Plotting with pcolormesh ---
# fig, ax = plt.subplots(figsize=(12, 6))

# U, A = np.meshgrid(u_dc_values, a_values)

# im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap, norm=norm)

# # Invert y-axis
# ax.invert_yaxis()

# # Add discrete colorbar with ticks at integers
# cbar = plt.colorbar(im, ax=ax, ticks=unique_vals)
# cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# # Overlay critical curve
# ax.plot(u_dc_values, a_crits, color="red")

# # Axis settings
# ax.set_ylim(-2, 4)
# ax.set_xlabel("u_dc", fontsize=16, fontweight="bold")
# ax.set_ylabel("a", fontsize=16, fontweight="bold")
# ax.set_title("Unique maxima count", fontsize=18, fontweight="bold")

# plt.tight_layout()
# plt.savefig(
#     f"unique-maxima-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-last-50000-values-pmesh.1.png",
#     dpi=300
# )
# plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.colors as mcolors
from joblib import Parallel, delayed

# Parameters
T = 500.0                      # Total simulation time (seconds)
t_rec = 3.0                    # Recording time (seconds)
noise = 2e-4                   # Noise magnitude
u_max = 1.0

omega_0 = 53956.46373431294
Q_0     = 50.0
alpha   = 19.2
beta    = 1066.0
gamma   = 1.62e7
R       = 12.5
tau     = 0.001
kappa   = 0.602e6

# a values
a_values = np.linspace(-3, 4, 1001)
a_values_flipped = a_values[::-1]   # reversed for downward scan

u_dc_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.11,
                        0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21, 0.22,
                        0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33,
                        0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44,
                        0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55,
                        0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66,
                        0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
                        0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88,
                        0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])

crossings = np.array([5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
                     5.0        , 5.0        , 3.98194408, 3.9197262 , 3.85942272,
                     3.80094662, 3.74421608, 3.68915407, 3.63568807, 3.58374967,
                     3.53327433, 3.48420107, 3.43647229, 3.39003347, 3.34483303,
                     3.30082206, 3.25795424, 3.2161856 , 3.17547439, 3.13578097,
                     3.09706761, 3.0592985 , 3.02243949, 2.98645806, 2.95132326,
                     2.91700555, 2.88347675, 2.85070997, 2.81867952, 2.78736086,
                     2.75673052, 2.72676605, 2.69744599, 2.66874976, 2.64065765,
                     2.6131508 , 2.5862111 , 2.5598212 , 2.53396442])

# Threshold for treating time series as constant
const_tol = 1e-8

# Preallocate
extrema_counts = np.zeros((len(a_values), len(u_dc_values)), dtype=int)

# --- Function for one u_dc ---
def process_u_dc_down(j, u_dc, a_values_flipped, const_tol, T, t_rec, u_max):
    results_down = np.load(
        f'/scratch/almo2783/scratch/dim-less/grid-search/deflections-down/'
        f'deflections-RK4-down-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-04-u_max-{u_max}-u_dc-{u_dc}.npy',
        mmap_mode="r"
    )
    
    col_results = np.zeros(len(a_values_flipped), dtype=int)

    for i, a in enumerate(a_values_flipped):
        data = results_down[i, :]

        if np.std(data) < const_tol:
            col_results[i] = 0
            continue

        data = data[-50000:]
        peaks, _ = find_peaks(data)
        maxima = data[peaks]
        unique_maxima = np.unique(maxima.round(decimals=4))

        if np.std(unique_maxima) < 0.01:
            col_results[i] = 1
        else:
            col_results[i] = len(unique_maxima)

    return j, col_results[::-1]

# --- Run in parallel ---
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_u_dc_down)(j, u_dc, a_values_flipped, const_tol, T, t_rec, u_max)
    for j, u_dc in enumerate(u_dc_values)
)

# --- Collect results ---
for j, col_results in results:
    extrema_counts[:, j] = col_results

# --- Critical curve ---
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
    a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc)) * (
        (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2))
        + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2)
        + (omega_0*omega_0**2/Q_0)*tau**2
        + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt(
            (omega_0/Q_0 + tau*omega_0**2)**2
            + (beta + beta*tau*omega_0/Q_0)**2
            + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0)
        )
    )

# --- Mask and Colormap ---
masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)
unique_vals = np.unique(extrema_counts[extrema_counts > 0])
colors = plt.cm.viridis_r(np.linspace(0, 1, len(unique_vals)))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color="black")

bounds = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
U, A = np.meshgrid(u_dc_values, a_values)

im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap, norm=norm)

# # Invert y-axis for downward scan
# ax.invert_yaxis()

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=unique_vals)
cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# Critical curve
ax.plot(u_dc_values, a_crits, color="red")
ax.plot(u_dc_values, crossings, color="red")

# Axes
ax.set_ylim(a_values.min(), a_values.max())
ax.set_xlabel("u_dc", fontsize=16, fontweight="bold")
ax.set_ylabel("a", fontsize=16, fontweight="bold")
ax.set_title("Unique maxima count (Downward)", fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig(
    f"unique-maxima-down-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-last-50000-values-pmesh-1.png",
    dpi=300
)
plt.close()
