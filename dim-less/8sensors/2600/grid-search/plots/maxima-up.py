import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
T = 50.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 2600.0, 16336.281798666923, 82.0, 0.0091, 109.89010989010988, 26827200.0, 50.24, 884700.0

a_values = np.linspace(-1, 1, 101)

u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
crossings = np.array([2.49495829, 1.24747925, 0.8316529 , 0.62374002, 0.49899218,
                      0.41582654, 0.35642278, 0.31187044, 0.2772186 , 0.24949654])
# Preallocate
extrema_counts = np.zeros((len(a_values), len(u_dc_values)), dtype=int)
const_tol = 1e-8

# --- Function to process one u_dc ---
def process_u_dc(j, u_dc, a_values, const_tol, T, t_rec, u_max):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/8sensors/2600/grid-search/deflections-up/'
        f'deflections-RK4-up-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-05-u_max-{u_max}-u_dc-{u_dc}.npy',
        mmap_mode="r"
    )
    
    col_results = np.zeros(len(a_values), dtype=int)

    for i, a in enumerate(a_values):
        data = results_up[i, :]

        if np.std(data) < const_tol:
            col_results[i] = 0
            continue

        # Take last part of signal
        data = data[-500000:]

        peaks, _ = find_peaks(data)
        maxima = data[peaks]
        unique_maxima = np.unique(maxima.round(decimals=4))

        if np.std(unique_maxima) < 0.01:
            col_results[i] = 1
        else:
            col_results[i] = len(unique_maxima)

    return j, col_results

# --- Run in parallel ---
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_u_dc)(j, u_dc, a_values, const_tol, T, t_rec, u_max)
    for j, u_dc in enumerate(u_dc_values)
)

# --- Reconstruct extrema_counts ---
for j, col_results in results:
    extrema_counts[:, j] = col_results

# calculating critical a for negative part
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
  a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

# # Mask out zeros (keep them black)
# masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

# # Use colormap and set bad values (masked ones) to black
# cmap = plt.cm.cool.copy()
# cmap.set_bad(color='black')

# # --- Plotting with pcolormesh ---
# fig, ax = plt.subplots(figsize=(12, 6))

# U, A = np.meshgrid(u_dc_values, a_values)

# im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

# # Add discrete colorbar with ticks at integers
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# # Overlay critical curve
# ax.plot(u_dc_values, a_crits, color="red")
# # ax.plot(u_dc_values, crossings, color="red")

# # Axis settings
# ax.set_ylim(a_values.min(), a_values.max())
# ax.set_xlabel("u_dc", fontsize=16, fontweight="bold")
# ax.set_ylabel("a", fontsize=16, fontweight="bold")
# ax.set_title("Unique maxima count (Upward)", fontsize=18, fontweight="bold")

# plt.tight_layout()
# plt.savefig(
#     f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-last-50000-values.png",
#     dpi=300
# )
# plt.close()

# Mask out zeros (keep them black)
masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)
# save extrema_counts
np.save('extrema_count_2600.npy', extrema_counts)
np.save('crossings_2600.npy', crossings)
np.save('a_crits_2600.npy', a_crits)
# Get unique values excluding 0
unique_vals = np.unique(extrema_counts[extrema_counts > 0])

# Build a discrete colormap (start from viridis but only pick needed number of colors)
colors = plt.cm.viridis_r(np.linspace(0, 1, len(unique_vals)))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color="black")  # for masked values (zeros)

# Normalize so each integer falls into its own color bin
bounds = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Plotting with pcolormesh ---
fig, ax = plt.subplots(figsize=(12, 6))

U, A = np.meshgrid(u_dc_values, a_values)

im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap, norm=norm)

# Add discrete colorbar with ticks at integers
cbar = plt.colorbar(im, ax=ax, ticks=unique_vals)
cbar.set_label("Number of unique maxima", fontsize=16, fontweight="bold")

# Overlay critical curve
ax.plot(u_dc_values, a_crits, color="red")
ax.plot(u_dc_values, crossings, color="red")

# Axis settings
ax.set_ylim(a_values.min(), a_values.max())
ax.set_xlabel("u_dc", fontsize=16, fontweight="bold")
ax.set_ylabel("a", fontsize=16, fontweight="bold")
ax.set_title("Unique maxima count (Upward)", fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig(
    f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-last-500000-values.png",
    dpi=300
)
plt.close()