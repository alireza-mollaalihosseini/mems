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

a_value = 0.44
u_dc_value = 0.4
lambda_value = 1e-3
mu = 1.0

# Load training, validation
X_train = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_train-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
X_val   = np.load(f"/scratch/almo2783/scratch/dim-less/8sensors/445/state-matrix/state_matrix_val-a-0.16-u_dc-0.1-mu-1.0.npz")['arr_0']
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Compute RF scores once
ranked_idx, rf_scores = compute_feature_ranking(X_train, labels_train)

# Run in parallel
np.save("idxes_gini.npy", ranked_idx)
np.save("scores_gini.npy", rf_scores)