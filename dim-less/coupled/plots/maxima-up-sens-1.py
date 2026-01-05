import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
T = 500.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

omega_0 = 53956.46373431294
Q_0     = 50.0
alpha   = 19.2
beta    = 1066.0
gamma   = 1.62e7
R       = 12.5
tau     = 0.001
kappa   = 0.602e6
u_dc    = 0.7

a_values = np.linspace(-3, 4, 1001)

c_f_values = np.array([-50.0, -48.0, -46.0, -44.0, -42.0, -40.0, -38.0, -36.0, -34.0, -32.0, -30.0,
                       -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0,  -8.0,
                        -6.0,  -4.0,  -2.0,   0.0,   2.0,   4.0,   6.0,   8.0,  10.0,  12.0,  14.0,
                        16.0,  18.0,  20.0,  22.0,  24.0,  26.0,  28.0,  30.0,  32.0,  34.0,  36.0,
                        38.0,  40.0,  42.0,  44.0,  46.0,  48.0,  50.0])

# crossings = np.array([5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 5.0        , 5.0        , 5.0        ,
#                      5.0        , 5.0        , 3.98194408, 3.9197262 , 3.85942272,
#                      3.80094662, 3.74421608, 3.68915407, 3.63568807, 3.58374967,
#                      3.53327433, 3.48420107, 3.43647229, 3.39003347, 3.34483303,
#                      3.30082206, 3.25795424, 3.2161856 , 3.17547439, 3.13578097,
#                      3.09706761, 3.0592985 , 3.02243949, 2.98645806, 2.95132326,
#                      2.91700555, 2.88347675, 2.85070997, 2.81867952, 2.78736086,
#                      2.75673052, 2.72676605, 2.69744599, 2.66874976, 2.64065765,
#                      2.6131508 , 2.5862111 , 2.5598212 , 2.53396442])

# Preallocate
extrema_counts = np.zeros((len(a_values), len(c_f_values)), dtype=int)
const_tol = 1e-8

# --- Function to process one u_dc ---
def process_c_f(j, c_f, a_values, const_tol, T, t_rec):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/coupled/deflections-up/'
        f'deflections-RK4-up-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-04-cf-{c_f}.npy',
        mmap_mode="r"
    )
    
    col_results = np.zeros(len(a_values), dtype=int)

    for i, a in enumerate(a_values):
        data = results_up[i, 0, :]

        if np.std(data) < const_tol:
            col_results[i] = 0
            continue

        # Take last part of signal
        data = data[-50000:]

        peaks, _ = find_peaks(data)
        maxima = data[peaks]
        if len(maxima) == 0:
            col_results[i] = 0
            continue
        unique_maxima = np.unique(maxima.round(decimals=4))

        if np.std(unique_maxima) < 0.01:
            col_results[i] = 1
        else:
            col_results[i] = len(unique_maxima)

    return j, col_results

# --- Run in parallel ---
results = Parallel(n_jobs=-1, backend="threading", verbose=5)(
    delayed(process_c_f)(j, c_f, a_values, const_tol, T, t_rec)
    for j, c_f in enumerate(c_f_values)
)

# --- Reconstruct extrema_counts matrix ---
extrema_counts = np.zeros((len(a_values), len(c_f_values)), dtype=int)
for j, col_results in results:
    extrema_counts[:, j] = col_results

# # calculating critical a for negative part
# a_crits = np.zeros(len(c_f_values))
# for i, u_dc in enumerate(c_f_values):
#   a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

# # Mask out zeros (keep them black)
# masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

# # Get unique values excluding 0
# unique_vals = np.unique(extrema_counts[extrema_counts > 0])

# # Build a discrete colormap (start from viridis but only pick needed number of colors)
# colors = plt.cm.viridis_r(np.linspace(0, 1, len(unique_vals)))
# cmap = mcolors.ListedColormap(colors)
# cmap.set_bad(color="black")  # for masked values (zeros)

# # Normalize so each integer falls into its own color bin
# bounds = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5, 1)
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# # --- Plotting with pcolormesh ---
# fig, ax = plt.subplots(figsize=(12, 6))

# U, A = np.meshgrid(c_f_values, a_values)

# im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap, norm=norm)

# # Add discrete colorbar with ticks at integers
# cbar = plt.colorbar(im, ax=ax, ticks=unique_vals)
# cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# # # Overlay critical curve
# # ax.plot(c_f_values, a_crits, color="red")
# # ax.plot(c_f_values, crossings, color="red")

# # Axis settings
# ax.set_ylim(a_values.min(), a_values.max())
# ax.set_xlabel("c_f", fontsize=16, fontweight="bold")
# ax.set_ylabel("a", fontsize=16, fontweight="bold")
# ax.set_title("Unique maxima count (Upward)", fontsize=18, fontweight="bold")

# plt.tight_layout()
# plt.savefig(
#     f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-sensor-1-last-50000-values.png",
#     dpi=300
# )
# plt.close()

# Mask out zeros (keep them black)
masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

# Use colormap and set bad values (masked ones) to black
cmap = plt.cm.viridis_r.copy()
cmap.set_bad(color='black')

# --- Plotting with pcolormesh ---
fig, ax = plt.subplots(figsize=(12, 6))

U, A = np.meshgrid(c_f_values, a_values)

im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# Axis settings
ax.set_ylim(a_values.min(), a_values.max())
ax.set_xlabel("c_f", fontsize=16, fontweight="bold")
ax.set_ylabel("a", fontsize=16, fontweight="bold")
ax.set_title("Unique maxima count (Upward)", fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig(
    f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-sensor-1-last-50000-values.png",
    dpi=300
)
plt.close()