import numpy as np

# Frequencies you actually simulated
freqs = np.linspace(1000, 50000, 100, dtype=int)

extrema_counts = {
    f: np.load(f"/scratch/almo2783/scratch/test/a-crit/extrems/extrema_counts_f_{f}-u-dc-1.0.npy")
    for f in freqs
}


a_values = np.linspace(0, 10000, 10001)
u_dc_values = [1.0]

a_crit = np.full((len(freqs)), np.nan)

for fi, f in enumerate(freqs):
    counts = extrema_counts[f]

    # for ui in range(counts.shape[1]):  # u_dc axis
    #     col = counts[:, ui]

    nz = np.nonzero(counts)[0]
    if nz.size > 0:
        a_crit[fi] = a_values[nz[0]]


np.save(f"/scratch/almo2783/scratch/test/a-crit/a-crits/a-crit-u-dc-1.0-more.npy", a_crit)