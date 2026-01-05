import numpy as np
import matplotlib.pyplot as plt


out_file = (
    "/scratch/almo2783/scratch/ml-paper/nonlinearity/full-timeseries/"
    "fft_frequency_ranking_train.npz"
)

data = np.load(out_file)
freq_axis = data["freq_axis"]                 # (n_freqs,)
freq_idx  = data["ranked_freq_indices"]       # (n_samples, n_freqs)
freq_std  = data["ranked_std"]                # (n_samples, n_freqs)
freq_mean = data["ranked_mean"]               # (n_samples, n_freqs)

# print(f"Shape of indexes: {freq_idx.shape}")
# print(f"Shape of std: {freq_std.shape}")
# print(f"Shape of mean: {freq_mean.shape}")

labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")

if labels_train.ndim == 2:
    labels = np.argmax(labels_train, axis=1)
else:
    labels = labels_train

N_WINDOWS = 10 
n_classes = 10
n_freqs   = freq_mean.shape[1]

class_mean = np.zeros((n_classes, n_freqs), dtype=np.float64)
class_std  = np.zeros((n_classes, n_freqs), dtype=np.float64)

for c in range(n_classes):
    idx = np.where(labels == c)[0]

    class_mean[c] = freq_mean[idx].mean(axis=0)
    class_std[c]  = freq_std[idx].mean(axis=0)

global_std = freq_std.mean(axis=0)
global_rank = np.argsort(global_std)[::-1]

TOPK = 1000  # Increased for wider range (~0–0.61 Hz); adjust as needed

# Use global ranking
top_freqs = freq_axis[global_rank[:TOPK]]  # Now ~0 to 0.61 Hz
ranked_class_mean_topk = class_mean[:, global_rank[:TOPK]]
ranked_class_std_topk = class_std[:, global_rank[:TOPK]]

# Optional: Exclude DC (bin 0) if unwanted
if top_freqs[0] == 0:
    top_freqs = top_freqs[1:TOPK]  # Drop first, shift slices accordingly
    ranked_class_mean_topk = ranked_class_mean_topk[:, 1:TOPK]
    ranked_class_std_topk = ranked_class_std_topk[:, 1:TOPK]
    TOPK -= 1

print(f"Top varying frequencies (Hz): {top_freqs[:5]} ... {top_freqs[-5:]}")  # Sample
print(f"Max freq in top K: {np.max(top_freqs)}")

for c in range(n_classes):
    plt.figure(figsize=(16, 8))

    plt.errorbar(
        top_freqs,
        ranked_class_mean_topk[c],
        yerr=ranked_class_std_topk[c],
        marker='o',
        linewidth=2,
        capsize=6,
        alpha=0.8,
        markersize=3  # Smaller markers for dense x
    )

    plt.xlabel("Frequency (Hz)", fontweight='bold', fontsize=20)
    plt.ylabel("Mean FFT Magnitude", fontweight='bold', fontsize=20)
    plt.title(f"Class {c} — Top {TOPK} Most Varying Frequencies over {N_WINDOWS} Windows\n(Range: {np.min(top_freqs):.2e}–{np.max(top_freqs):.3f} Hz)",
              fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=16)
    plt.yticks(fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)

    # Log x for narrow low-freq range (avoids log(0))
    plt.xscale('log')
    plt.xlim(np.min(top_freqs) * 0.5, np.max(top_freqs) * 2)

    plt.tight_layout()
    plt.savefig(f"class_{c}_freq_variation.png", dpi=300, bbox_inches='tight')
    plt.close()



# N_TOP = 50

# plt.figure(figsize=(16,8))

# for c in range(n_classes):
#     plt.errorbar(
#         np.arange(N_TOP),
#         ranked_class_mean[c, :N_TOP],
#         yerr=ranked_class_std[c, :N_TOP],
#         linewidth=2,
#         capsize=5,
#         label=f"Class {c}"
#     )

# plt.xlabel("Ranked frequency index", fontweight='bold', fontsize=20)
# plt.ylabel("FFT Magnitude", fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=16)
# plt.yticks(fontweight='bold', fontsize=16)

# plt.legend(fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()
# plt.savefig("class_avg_top50_ranked_frequencies.png", dpi=300)
# plt.close()


# for c in range(n_classes):
#     plt.figure(figsize=(16,8))
    
#     plt.errorbar(
#         np.arange(N_TOP),
#         ranked_class_mean[c, :N_TOP],
#         yerr=ranked_class_std[c, :N_TOP],
#         marker='o',
#         linewidth=2,
#         capsize=6,
#         label=f"Class {c}"
#     )

#     plt.xlabel("Ranked frequency index", fontweight='bold', fontsize=20)
#     plt.ylabel("FFT Magnitude", fontweight='bold', fontsize=20)
#     plt.legend(fontsize=16)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.8)
#     plt.tight_layout()

#     plt.savefig(f"class_{c}_top50_ranked_freqs.png", dpi=300)
#     plt.close()




# sample = 23   # choose any index

# plt.figure(figsize=(16,8))

# # class average curve of sample's true class
# c = labels[sample]

# plt.errorbar(
#     np.arange(N_TOP),
#     ranked_class_mean[c,:N_TOP],
#     yerr=ranked_class_std[c,:N_TOP],
#     linewidth=2,
#     capsize=4,
#     label="Class avg"
# )

# # sample points
# plt.scatter(
#     np.arange(N_TOP),
#     freq_mean[sample, ranked_freq[:N_TOP]],
#     color='r',
#     s=40,
#     label="Single sample"
# )

# plt.legend(fontsize=16)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("sample_vs_class_avg.png", dpi=300)
# plt.close()







# sample = 23

# plt.figure(figsize=(16,8))
# # plt.errorbar(
# #     freq_idx, freq_mean,
# #     yerr=freq_std,
# #     marker='o',
# #     linewidth=2,
# #     capsize=6
# # )

# plt.plot(freq_idx[sample, :1000], freq_mean[sample, :1000], label="magnitutes")

# plt.scatter(freq_idx[sample, :50], freq_mean[sample, :50], color='r', label="Top 50")

# # Formatting
# plt.xlabel("Frequencies", fontweight='bold', fontsize=20)
# plt.ylabel("Magnitudes", fontweight='bold', fontsize=20)
# plt.xticks(fontweight='bold', fontsize=20)
# plt.yticks(fontweight='bold', fontsize=20)
# plt.legend(fontsize=18)
# plt.grid(True, which='both', linestyle='--', linewidth=0.8)
# plt.tight_layout()

# plt.savefig("sorted_freq_idx.png", dpi=300)
# plt.close()