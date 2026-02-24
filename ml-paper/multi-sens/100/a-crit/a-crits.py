import numpy as np

# Frequencies you actually simulated
freqs = np.linspace(1000, 50000, 100, dtype=int)

extrema_counts = {
    f: np.load(f"/scratch/almo2783/scratch/test/a-crit/extrems/extrema_counts_f_{f}-more.npy")
    for f in freqs
}

# a_values = np.linspace(0, 1500, 501)
a_values = np.linspace(0, 1500, 1501)
u_dc_values = np.linspace(0.1, 1.0, 10)

a_crit = np.full((len(freqs), len(u_dc_values)), np.nan)

for fi, f in enumerate(freqs):
    counts = extrema_counts[f]

    for ui in range(counts.shape[1]):  # u_dc axis
        col = counts[:, ui]

        nz = np.nonzero(col)[0]
        if nz.size > 0:
            a_crit[fi, ui] = a_values[nz[0]]


for i in range(a_crit.shape[1]):
  np.save(f"/scratch/almo2783/scratch/test/a-crit/a-crits/a-crit-u-dc-{u_dc_values[i]:.1f}-more.npy", a_crit[:, i])