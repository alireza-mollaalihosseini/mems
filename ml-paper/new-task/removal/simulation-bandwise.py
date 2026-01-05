from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Ridge prediction
# ------------------------------------------------------------
def ridge_predict(X, W):
    """
    X : (n_samples, n_features)
    W : (n_features + 1, n_classes)
    """
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])  # add bias column
    scores = Xb @ W
    return np.argmax(scores, axis=1)


# ------------------------------------------------------------
# Helper: remove feature band from weight matrix
# ------------------------------------------------------------
def remove_band_from_weights(W, start, end):
    """
    Removes rows [start:end] from the W matrix (weights), 
    but preserves the last row (bias).
    """
    W_no_bias = W[:-1]      # all weights except bias
    W_bias = W[-1:]         # bias row

    # remove feature band
    W_reduced = np.delete(W_no_bias, np.s_[start:end], axis=0)

    # add bias row back
    return np.vstack([W_reduced, W_bias])


# ------------------------------------------------------------
# Band-wise removal importance
# ------------------------------------------------------------
def removal_importance_bands(
    X, y_true, W, band_size=20, n_jobs=-1
):
    """
    Band-wise removal importance for FFT-based linear ridge model.
    Removes each band entirely (no permutation).

    X : validation data (samples × features)
    y_true : one-hot labels
    W : ridge weight matrix
    band_size : number of adjacent bins per band
    """

    y_true_int = np.argmax(y_true, axis=1)

    # Baseline accuracy
    baseline_acc = accuracy_score(y_true_int, ridge_predict(X, W))

    n_samples, n_features = X.shape
    n_bands = n_features // band_size

    # Compute band indices
    bands = [
        (i * band_size, min((i + 1) * band_size, n_features))
        for i in range(n_bands)
    ]

    # Worker function
    def evaluate_band(b):
        start, end = b

        # remove features from X
        X_removed = np.delete(X, np.s_[start:end], axis=1)

        # remove same features from W
        W_removed = remove_band_from_weights(W, start, end)

        # accuracy after removal
        acc = accuracy_score(
            y_true_int,
            ridge_predict(X_removed, W_removed)
        )

        # importance = accuracy drop
        return baseline_acc - acc

    # Parallel execution
    importance = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_band)(band) for band in bands
    )

    return np.array(importance), bands


# ------------------------------------------------------------
# Top-k utility
# ------------------------------------------------------------
def top_k_frequencies(importance, centers, k=10):
    idx = np.argsort(importance)[::-1][:k]
    return centers[idx], importance[idx]


# ------------------------------------------------------------
# --- Usage Example ---
# ------------------------------------------------------------

# Load data
X      = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']
X_val  = np.load('/scratch/almo2783/scratch/ml-paper/new-task/state-matrix/state_matrix_val-a-0.44-u_dc-0.4-mu-1.0.npz')['arr_0']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_val_std = scaler.transform(X_val)

labels_train = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_train.npy")
labels_val   = np.load("/scratch/almo2783/scratch/ml-paper/new-task/filepaths/label_matrix_val.npy")

weights = np.load('/scratch/almo2783/scratch/ml-paper/new-task/weights/weights-a-0.44-lambda-0.001.npz')['arr_0']

# Parameters
band_size = 20
X_val = X_val_std
y_val = labels_val
W = weights

# Frequency axis (for full FFT bins)
freqs = np.linspace(0, 24000, X_val.shape[1])

# Compute band removal importance
importance, bands = removal_importance_bands(
    X_val, y_val, W,
    band_size=band_size,
    n_jobs=64
)

# Save results
np.save("bandwise_importance.npy", importance)
np.save("bands.npy", bands)

# Compute band center frequencies
centers = np.array([(s + e) / 2 for (s, e) in bands])

# Plot
plt.figure(figsize=(16, 8))
plt.plot(centers, importance, linewidth=2)
plt.title("Band-wise Removal Importance", fontsize=20)
plt.xlabel("Frequency Bin (center of band)", fontsize=20)
plt.ylabel("Accuracy Drop", fontsize=20)
plt.grid(True)
plt.savefig('importance-band-wise.png', dpi=300)
plt.close()

# Top-k
top_freqs, top_vals = top_k_frequencies(importance, centers, k=100)
print("Top frequencies:", top_freqs)
print("Importance values:", top_vals)
