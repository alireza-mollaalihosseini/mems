from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --------------------------------------
# Ridge prediction
# --------------------------------------
def ridge_predict(X, W):
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    scores = Xb @ W
    return np.argmax(scores, axis=1)


# --------------------------------------
# Band-wise permutation importance
# --------------------------------------
def permutation_importance_bands(
    X, y_true, W, band_size=20, n_repeats=5, n_jobs=-1
):
    """
    Band-wise permutation importance for FFT-based model.

    X : validation data
    y_true : one-hot labels
    W : ridge weights
    band_size : number of adjacent bins
    n_repeats : times to shuffle each band
    """

    y_true_int = np.argmax(y_true, axis=1)

    # Baseline accuracy
    baseline_acc = accuracy_score(y_true_int, ridge_predict(X, W))

    n_samples, n_features = X.shape
    n_bands = n_features // band_size

    # Precompute band indices
    bands = [
        (i * band_size, min((i + 1) * band_size, n_features))
        for i in range(n_bands)
    ]

    # Worker function
    def evaluate_band(b):
        start, end = b
        drops = []

        X_temp = X.copy()

        for _ in range(n_repeats):
            # shuffle whole band
            for col in range(start, end):
                np.random.shuffle(X_temp[:, col])

            acc = accuracy_score(
                y_true_int, ridge_predict(X_temp, W)
            )
            drops.append(baseline_acc - acc)

            # restore original band
            X_temp[:, start:end] = X[:, start:end]

        return np.mean(drops)

    # Parallel computation
    importance = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_band)(band) for band in bands
    )

    return np.array(importance), bands


def top_k_frequencies(importance, freqs, k=10):
    idx = np.argsort(importance)[::-1][:k]
    return freqs[idx], importance[idx]



X      = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
X_val  = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_val_std  = scaler.transform(X_val)

labels_train = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_train.npy")
labels_val = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_val.npy")
y = np.argmax(labels_train, axis=1)

weights = np.load('/scratch/almo2783/scratch/ml-paper/new-task/weights/weights-a-0.44-lambda-0.001.npz')['arr_0']

band_size = 20


# Load data as usual

X_val = X_val_std       # (num_samples, n_features)
y_val = labels_val          # one-hot labels
W = weights                   # loaded from your npz

# Frequency axis for plotting
freqs = np.linspace(0, 24000, X_val.shape[1])

importance, bands = permutation_importance_bands(
    X_val, y_val, W,
    band_size=band_size,
    n_repeats=5,
    n_jobs=64
)

# save importance
np.save("bandwise_importance.npy", importance)
np.save("bands.npy", bands)

# plot importance
centers = [(s + e)/2 for (s, e) in bands]

plt.figure(figsize=(16, 8))
plt.plot(centers, importance, linewidth=2)
plt.title("Band-wise permutation")
plt.xlabel("Frequency Bin (center of band)")
plt.ylabel("Accuracy Drop")
plt.grid(True)
plt.savefig('importance-band-wise.png', dpi=300)
plt.close()

top_freqs, top_vals = top_k_frequencies(importance, freqs, k=100)
print("Top frequencies:", top_freqs)
print("Importance values:", top_vals)