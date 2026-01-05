import numpy as np
import matplotlib.pyplot as plt
import os

def scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

sim_dir = "/scratch/almo2783/scratch/ml-paper/nonlinearity/results-avg"
a_values = np.linspace(-1, 1, 101)
rho_all, mi_all, nlr_all = [], [], []
u_dc = 0.4

for a in a_values:
    sim_path_rho = os.path.join(sim_dir, f"avg_rho_a_{a:.2f}.npz")
    sim_path_mi = os.path.join(sim_dir, f"avg_mi_a_{a:.2f}.npz")
    sim_path_nlr = os.path.join(sim_dir, f"avg_nlr_a_{a:.2f}.npz")

    if not os.path.exists(sim_path_rho or sim_path_mi or sim_path_nlr):
        print(f"⚠️ Missing file for a={a:.2f}")
        correlations.append(np.nan)
        continue

    rho = np.load(sim_path_rho)['rho']
    mi  = np.load(sim_path_mi)['rho']
    nlr = np.load(sim_path_nlr)['rho']

    rho_all.append(rho)
    mi_all.append(mi)
    nlr_all.append(nlr)

# save data
np.save('average-rho.npy', rho_all)
np.save('average-mi.npy', mi_all)
np.save('average-nlr.npy', nlr_all)

# load validations
val_acc = np.load('/scratch/almo2783/scratch/ml-paper/nonlinearity/plots/val_accuracies.npy')
time_val_acc = np.load('/scratch/almo2783/scratch/ml-paper/time-domain/fixed-u_dc/plots/val_acc.npy')
time_val_acc = time_val_acc / 100

rho_all = scale(rho_all)
mi_all = scale(mi_all)
nlr_all = scale(nlr_all)
# val_acc = scale(val_acc)
# time_val_acc = scale(time_val_acc)

plt.figure(figsize=(16,8))
plt.plot(a_values, rho_all, marker="o", linewidth=2, label='rho')
plt.plot(a_values, mi_all, marker="o", linewidth=2, label='MI')
plt.plot(a_values, nlr_all, marker="o", linewidth=2, label='NLR')
plt.plot(a_values, val_acc, marker="o", linewidth=2, label='Val Acc.')
plt.plot(a_values, time_val_acc, marker="o", linewidth=2, label='Time Val Acc.')
plt.axvline(x=0.44, color='r', linestyle='--', alpha=0.7)
plt.title(f"Linearity/Nonlinearity vs a (u_dc = {u_dc})", fontsize=20, fontweight='bold')
plt.xlabel("a values", fontsize=20, fontweight='bold')
plt.ylabel("Scales", fontsize=20, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontweight='bold', fontsize=18)
plt.yticks(fontweight='bold', fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig('Average_corr.png', dpi=300)
plt.close()
