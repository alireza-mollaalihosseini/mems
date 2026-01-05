import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from joblib import Parallel, delayed

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

a_values = np.linspace(-3, 4, 1001)

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

# Sampling frequency
fs = 1.0 / 1e-6

# Preallocate
peak_freqs = np.zeros((len(a_values), len(u_dc_values)))
const_tol = 1e-8

# --- Function for parallel computation ---
def process_one(j, u_dc):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/grid-search/deflections-up/'
        f'deflections-RK4-up-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-2e-04-u_max-{u_max}-u_dc-{u_dc}.npy',
        mmap_mode="r"
    )
    col_result = np.zeros(len(a_values))
    for i, a in enumerate(a_values):
        signal = results_up[i, :]

        if np.std(signal) < const_tol:
            col_result[i] = -1
            continue

        freqs, psd = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
        peak_idx = np.argmax(psd)
        col_result[i] = freqs[peak_idx]

    return j, col_result

# --- Run parallel loop ---
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_one)(j, u_dc) for j, u_dc in enumerate(u_dc_values)
)

# --- Collect results ---
for j, col_result in results:
    peak_freqs[:, j] = col_result


# calculating critical a for negative part
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
  a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )


# Mask out negative ones
masked_data = np.ma.masked_where(peak_freqs == -1, peak_freqs)

# Use colormap and set bad values (masked ones) to black
cmap = plt.cm.cool.copy()
cmap.set_bad(color='black')

# --- Plot heatmap ---
U, A = np.meshgrid(u_dc_values, a_values)
fig, ax = plt.subplots(figsize=(12, 6))
pcm = ax.pcolormesh(
    U, A, masked_data,
    shading="nearest", cmap=cmap
)
plt.colorbar(pcm, label="Peak frequency [Hz]")
# Overlay a_crits curve
ax.plot(u_dc_values, a_crits, color='green')
ax.plot(u_dc_values, crossings, color="red")

ax.set_ylim(-3, 4)
ax.set_xlabel("u_dc", fontsize=14, fontweight="bold")
ax.set_ylabel("a_value", fontsize=14, fontweight="bold")
ax.set_title("Peak frequency heatmap", fontsize=16, fontweight="bold")
plt.savefig("Peak-frequency-heatmap.png")
plt.close()
