import numpy as np
from sklearn.feature_selection import f_classif

mu = 1.0
sensors = np.array([445, 490, 582, 591, 1109, 1161, 2600])
y = np.load('/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy')
best_values = {
    445: {'a': 0.16, 'u_dc': 0.1},
    490: {'a': 0.04, 'u_dc': 0.9},
    582: {'a': 0.02, 'u_dc': 0.9},
    591: {'a': 0.60, 'u_dc': 0.1},
    1109: {'a': 0.06, 'u_dc': 1.0},
    1161: {'a': 0.44, 'u_dc': 0.1},
    2600: {'a': 0.44, 'u_dc': 0.4},
}

for sensor in sensors:

    a = best_values[sensor]['a']
    u_dc = best_values[sensor]['u_dc']
    X = np.load(f'/scratch/almo2783/scratch/dim-less/8sensors/{sensor}/state-matrix/state_matrix_train-a-{a}-u_dc-{u_dc}-mu-1.0.npz')['arr_0']

    all_top_idx = []

    for i in range(y.shape[1]):
        # Compute F-scores per feature
        F, p = f_classif(X, y[:, i])
        # anova_scores[sensor] = F

        # Plot top-k features
        top_k = 500
        top_idx = np.argsort(F)[::-1][:top_k]
        all_top_idx.append(top_idx)

    unique_top_idx = np.unique(np.concatenate(all_top_idx))
    length = len(unique_top_idx)

    np.save(f'/scratch/almo2783/scratch/dim-less/8sensors/feature-par/anova-test/indices/feature_indices_sensor_{sensor}_top_{top_k}_length_{length}.npy', unique_top_idx)