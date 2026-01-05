from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def ridge_predict(X, W):
    """
    Predict class labels using ridge regression closed-form solution.
    X: samples x features
    W: (features+1) x classes, last row = bias
    """
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    scores = Xb @ W
    return np.argmax(scores, axis=1)


def _permute_feature_drop(f, X, y_true_int, W, baseline_acc, n_repeats):
    """
    Worker function for a single feature.
    """
    X_perm = X.copy()
    drops = []

    for _ in range(n_repeats):
        # Shuffle only column f
        np.random.shuffle(X_perm[:, f])

        # Predict
        y_pred_perm = ridge_predict(X_perm, W)
        acc = accuracy_score(y_true_int, y_pred_perm)

        drops.append(baseline_acc - acc)

        # Restore original column
        X_perm[:, f] = X[:, f]

    return np.mean(drops)


def permutation_importance_ridge(X, y_true, W, n_repeats=5, n_jobs=-1):
    """
    Parallelized permutation importance for closed-form ridge regression.
    """
    # Convert one-hot → int labels
    y_true_int = np.argmax(y_true, axis=1)

    # Baseline accuracy
    baseline_acc = accuracy_score(y_true_int, ridge_predict(X, W))

    n_features = X.shape[1]

    # Parallel execution for all features
    importance = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(
        delayed(_permute_feature_drop)(
            f, X, y_true_int, W, baseline_acc, n_repeats
        )
        for f in range(n_features)
    )

    return np.array(importance)


def top_k_frequencies(importance, freqs, k=10):
    idx = np.argsort(importance)[::-1][:k]
    return freqs[idx], importance[idx]


def plot_importance_curve(importance, freqs):
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, importance, linewidth=2)
    plt.title("Permutation Feature Importance (Closed-Form Ridge)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Accuracy Drop")
    plt.grid(True)
    plt.show()



X      = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
X_val  = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_val_std  = scaler.transform(X_val)

labels_train = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_train.npy")
labels_val = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_val.npy")
y = np.argmax(labels_train, axis=1)

weights = np.load('/scratch/almo2783/scratch/ml-paper/new-task/weights/weights-a-0.44-lambda-0.001.npz')['arr_0']


# Load data as usual

X_val = X_val_std       # (num_samples, n_features)
y_val = labels_val          # one-hot labels
W = weights                   # loaded from your npz

# Frequency axis for plotting
freqs = np.linspace(0, 24000, X_val.shape[1])

importance = permutation_importance_ridge(X_val, y_val, W, n_repeats=5, n_jobs=64)

# save importance
np.save('Importance.npy', importance)

# plot importance
plt.figure(figsize=(12, 5))
plt.plot(freqs, importance, linewidth=2)
plt.title("Permutation Feature Importance")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Accuracy Drop")
plt.grid(True)
plt.savefig('importance.png', dpi=300)
plt.close()

top_freqs, top_vals = top_k_frequencies(importance, freqs, k=100)
print("Top frequencies:", top_freqs)
print("Importance values:", top_vals)