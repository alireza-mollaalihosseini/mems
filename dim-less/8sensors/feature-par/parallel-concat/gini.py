import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def compute_feature_ranking(X, y):
    """
    Compute feature ranking using a Random Forest classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features).
    y : np.ndarray
        One-hot encoded or label vector.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 uses all cores).

    Returns
    -------
    ranked_idx : np.ndarray
        Indices of features sorted by importance (descending).
    importances : np.ndarray
        Importance score for each feature.
    """
    # Ensure we have class labels
    y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

    rf = RandomForestClassifier(
            n_estimators=5000,          # sweet spot: max stability without insanity
            max_depth=None,             # let trees grow fully (important for pure Gini importance)
            min_samples_leaf=1,         # no regularization → purest possible importance
            min_samples_split=2,
            max_features=0.5,           # or 'sqrt' — 0.5 often gives slightly more stable rankings in very high dim
            bootstrap=True,             # must be True for proper out-of-bag based importance
            oob_score=True,             # optional sanity check
            random_state=42,
            n_jobs=64
        )
    rf.fit(X, y_labels)

    # Extract feature importances
    importances = rf.feature_importances_
    ranked_idx = np.argsort(importances)[::-1]  # descending

    return ranked_idx, importances

# Load labels
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")

# --- Load full matrices ONCE ---
X_train1_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0'][:,:5000]
X_train2_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/490/state-matrix/state_matrix_train-a-0.04-u_dc-0.9-mu-1.0.npz")['arr_0'][:,:5000]
X_train3_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/582/state-matrix/state_matrix_train-a-0.02-u_dc-0.9-mu-1.0.npz")['arr_0'][:,:5000]
X_train4_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/591/state-matrix/state_matrix_train-a-0.6-u_dc-0.1-mu-1.0.npz")['arr_0'][:,:6000]
X_train5_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1109/state-matrix/state_matrix_train-a-0.06-u_dc-1.0-mu-1.0.npz")['arr_0'][:,:16000]
X_train6_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/1161/state-matrix/state_matrix_train-a-0.44-u_dc-0.1-mu-1.0.npz")['arr_0'][:,:16000]
X_train7_full  = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/2600/state-matrix/state_matrix_train-a-0.44-u_dc-0.4-mu-1.0.npz")['arr_0'][:,:16000]


# concatenate all states together
X_train = np.concatenate([X_train1_full, X_train2_full, X_train3_full, X_train4_full, X_train5_full, X_train6_full, X_train7_full], axis=1)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Compute RF scores once
ranked_idx, rf_scores = compute_feature_ranking(X_train, labels_train)

# Run in parallel
np.save("idxes_gini_concat.npy", ranked_idx)
np.save("scores_gini_concat.npy", rf_scores)