import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# load scores and indexes
scores = np.load("feature_importances.npy")
idx = np.load("feature_ranking_idx.npy")

# plot the gini scores distribution
plt.figure(figsize=(16,8))
# plt.plot(idx, scores, marker='o', linestyle='-', markersize=2)
# plt.scatter(idx, scores, s=15)
plt.plot(scores, "-o", label="Gini Scores")
plt.axhline(25*np.mean(scores), color="blue", linestyle='--', label="25*avg(scores)")
plt.plot(
    idx[:50],
    np.sort(scores)[::-1][:50],
    'o',
    markersize=10,
    markeredgewidth=1,
    markerfacecolor='none',
    label="top 50 scores"
)
plt.title("Gini Importance Scores Distribution", fontsize=16)
plt.xlabel("Feature Index", fontsize=20)
plt.ylabel("Gini Importance Score", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("gini_importance_distributionn.png", dpi=300)
plt.close()