import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
T = 499.0
t_rec = 2.0
noise = 2e-5
u_max = 1.0

omega_0 = 88485.45437478434
Q_0     = 43.2
alpha   = 749.37
beta    = 1066.0
gamma   = 4.2588e7
R       = 25.0
tau     = 0.001
kappa   = 1e6

a_values = np.linspace(-2, 2, 1001)

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
                      5.0        , 5.0        , 1.97966409, 1.75970141, 1.58373127,
                      1.4397557 , 1.31977606, 1.21825483, 1.13123663, 1.05582085,
                      0.98983205, 0.93160663, 0.87985071, 0.83354278, 0.79186564,
                      0.75415775, 0.71987785, 0.68857882, 0.65988803, 0.63349252,
                      0.60912742, 0.58656714, 0.56561832, 0.54611424, 0.52791042,
                      0.51088106, 0.49491603, 0.47991857, 0.46580332, 0.45249465,
                      0.43992535, 0.42803548, 0.41677139, 0.40608495, 0.39593282,
                      0.38627593, 0.37707888, 0.3683096 , 0.35993893, 0.35194028,
                      0.34428941, 0.33696411, 0.32994403, 0.32321047, 0.31674626,
                      0.31053556, 0.30456371, 0.29881723, 0.29328358, 0.28795114,
                      0.28280917, 0.27784761, 0.27305713, 0.26842904, 0.26395521,
                      0.25962808, 0.25544054, 0.25138593, 0.24745802, 0.24365097,
                      0.23995928, 0.23637781, 0.23290167, 0.22952629, 0.22624734,
                      0.22306076, 0.21996268, 0.2169495 , 0.21401776, 0.21116418,
                      0.2083857 , 0.20567941, 0.20304249, 0.20047232, 0.19796643,
                      0.19552239, 0.19313798, 0.19081102, 0.18853945, 0.18632135,
                      0.1841548 , 0.1820381 , 0.17996946, 0.17794736, 0.17597014,
                      0.17403643, 0.17214471, 0.17029371, 0.16848206, 0.16670858,
                      0.16497203, 0.16327128, 0.16160526, 0.15997286])

# Preallocate
extrema_counts = np.zeros((len(a_values), len(u_dc_values)), dtype=int)
const_tol = 1e-8

# --- Function to process one u_dc ---
def process_u_dc(j, u_dc, a_values, const_tol, T, t_rec, u_max):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/grid-search/design1/deflections-up/'
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
        data = data[-50000:]

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

# print(np.unique(extrema_counts))
# calculating critical a for negative part
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
  a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )

# Mask out zeros (keep them black)
masked_data = np.ma.masked_where(extrema_counts == 0, extrema_counts)

# Use colormap and set bad values (masked ones) to black
cmap = plt.cm.cool.copy()
cmap.set_bad(color='black')

# --- Plotting with pcolormesh ---
fig, ax = plt.subplots(figsize=(12, 6))

U, A = np.meshgrid(u_dc_values, a_values)

im = ax.pcolormesh(U, A, masked_data, shading="nearest", cmap=cmap)

# Add discrete colorbar with ticks at integers
cbar = plt.colorbar(im, ax=ax)
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
    f"unique-maxima-up-t-sim-{T}-t-rec-{t_rec}-noise-{noise:.0e}-u_max-{u_max}-last-50000-values.png",
    dpi=300
)
plt.close()