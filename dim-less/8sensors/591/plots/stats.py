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
crossings = np.array([1.31108922, 0.65554874, 0.43706658, 0.32783228, 0.26229155,
                      0.21859287, 0.18735247, 0.163968  , 0.14581758, 0.13122251])
lam = 1e4
mu = 1.0
typs = ['test', 'val']

# Parameters
T = 50.0
t_rec = 3.0
noise = 2e-4
u_max = 1.0

# Simulation params
alpha, f, omega_0, Q_0, tau, beta, gamma, R, kappa = 12.5, 591.0, 3713.3625165431354, 45.0, 0.0039000000000000003, 256.4102564102564, 1800000.0, 13.4, 214190.0

# calculating critical a for negative part
a_crits = np.zeros(len(u_dc_values))
for i, u_dc in enumerate(u_dc_values):
  a_crits[i] = (-R**2/(4*gamma*alpha*tau**2*kappa*u_dc))*( (beta+beta**2*tau + (omega_0/Q_0)*(1+beta*tau+beta**2*tau**2)) + (omega_0**2/Q_0)*(1/Q_0 - Q_0)*(tau + beta*tau**2) + (omega_0*omega_0**2/Q_0)*tau**2 + (1+beta*tau+ (tau*omega_0/Q_0))*np.sqrt( (omega_0/Q_0 + tau*omega_0**2)**2 + (beta + beta*tau*omega_0/Q_0)**2 + 2*beta*omega_0*(-tau*omega_0 + 1/Q_0 + tau*omega_0/Q_0**2 + tau**2*omega_0**2/Q_0) ) )


# Worker function to load data
def process_file(a, u_dc, typ, mu, lam):
    file_name = f'/scratch/almo2783/scratch/dim-less/8sensors/591/results/results_{typ}-a-{a}-u_dc-{u_dc}-mu-{mu}-lam-{lam:.0e}.txt'
    if os.path.exists(file_name):
        try:
            data = np.loadtxt(file_name)
            return (typ, a, u_dc, data[4] * 100)  # test accuracy
        except Exception:
            return (typ, a, u_dc, np.nan)
    else:
        return (typ, a, u_dc, np.nan)

# Run in parallel
results = Parallel(n_jobs=-1)(
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

    # ---- Find the 10 highest values ----
    flat_indices = np.argpartition(data_matrix.ravel(), -10)[-10:]
    flat_indices = flat_indices[np.argsort(data_matrix.ravel()[flat_indices])[::-1]]  # sort descending

    top10_results = []
    for flat_idx in flat_indices:
        row, col = np.unravel_index(flat_idx, data_matrix.shape)
        val = data_matrix[row, col]
        a_val = a_values[row]
        u_dc_val = u_dc_values[col]

        # Distances from both bifurcation curves
        dist_cross = np.abs(a_val - crossings[col])
        dist_acrit = np.abs(a_val - a_crits[col])

        top10_results.append({
            "accuracy": float(val),
            "a": float(a_val),
            "u_dc": float(u_dc_val),
            "dist_crossing": float(dist_cross),
            "dist_a_crit": float(dist_acrit)
        })

    # ---- Save results ----
    save_path = f"top10_{typ}_results.npy"
    np.save(save_path, top10_results, allow_pickle=True)
    # print(f"Saved top 10 results for {typ} → {save_path}")

    # Optional: also save as CSV for readability
    import csv
    csv_path = f"top10_{typ}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=top10_results[0].keys())
        writer.writeheader()
        writer.writerows(top10_results)
    # print(f"Saved top 10 results for {typ} → {csv_path}")

