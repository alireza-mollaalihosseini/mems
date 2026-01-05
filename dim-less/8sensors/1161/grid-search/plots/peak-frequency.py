import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from joblib import Parallel, delayed

# Parameters
T = 50.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 1161.0, 7294.778141635499, 57.0, 0.0057, 175.43859649122805, 2391680.0, 13.23, 361630.0

a_values = np.linspace(-1, 1, 101)

u_dc_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
crossings = np.array([1.51056986, 0.75528798, 0.5035284 , 0.37764874, 0.3021152 ,
                      0.25176933, 0.21580772, 0.18882988, 0.16785022, 0.15107548])
# Sampling frequency
fs = 1.0 / 1e-6

# Preallocate
peak_freqs = np.zeros((len(a_values), len(u_dc_values)))
const_tol = 1e-8

# --- Function for parallel computation ---
def process_one(j, u_dc):
    results_up = np.load(
        f'/scratch/almo2783/scratch/dim-less/8sensors/1161/grid-search/deflections-up/'
        f'deflections-RK4-up-t-sim-{int(T)}-t-rec-{int(t_rec)}-noise-5e-06-u_max-{u_max}-u_dc-{u_dc}.npy',
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
fig, ax = plt.subplots(figsize=(12, 6))
U, A = np.meshgrid(u_dc_values, a_values)
pcm = ax.pcolormesh(
    U, A, masked_data,
    shading="nearest", cmap=cmap
)
# plt.colorbar(pcm, label="Peak frequency [Hz]")
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("Peak frequency [Hz]", fontsize=16, fontweight="bold")

# Overlay a_crits curve
ax.plot(u_dc_values, a_crits, color='green')
ax.plot(u_dc_values, crossings, color="red")

# ax.set_ylim(-3, 4)
ax.set_ylim(a_values.min(), a_values.max())
ax.set_xlabel("u_dc", fontsize=16, fontweight="bold")
ax.set_ylabel("a_value", fontsize=16, fontweight="bold")
ax.set_title("Peak frequency heatmap", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("Peak-frequency-heatmap.png", dpi=300)
plt.close()
