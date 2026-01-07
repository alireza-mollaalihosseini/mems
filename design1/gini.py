import sys
import time
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


a_value = 0.6
u_dc_value = 0.1
mu = 1.0

# Load training, validation
X_train = np.load(f"/scratch/almo2783/scratch/design1/state-matrices/train_state_a_0.60_u_dc_0.10.npz")['arr_0']
X_val   = np.load(f"/scratch/almo2783/scratch/design1/state-matrices/val_state_a_0.60_u_dc_0.10.npz")['arr_0']
labels_train = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_train.npy")
labels_val   = np.load(f"/scratch/almo2783/scratch/dim-less/barcelona/label_matrix_val.npy")

# Convert labels to class indices (0..9)
y_train = np.argmax(labels_train, axis=1) if labels_train.ndim > 1 else labels_train
y_val   = np.argmax(labels_val, axis=1) if labels_val.ndim > 1 else labels_val

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

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
rf.fit(X_train, y_train)

# Predict on validation set
y_pred = rf.predict(X_val)

acc = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')

print(f"Validation accuracy: {acc:.4f}")
print(f"Precision (macro):   {precision:.4f}")
print(f"Recall (macro):      {recall:.4f}")
print(f"F1-score (macro):    {f1:.4f}")
print("Confusion matrix:")

# Extract feature importances
rf_scores = rf.feature_importances_
ranked_idx = np.argsort(importances)[::-1]  # descending

# Save model and preprocessing
dump(rf_clf, "rf_classifier.joblib")
dump(scaler, "scaler.joblib")

# Save feature ranking (already done, but grouped for clarity)
np.save("feature_ranking_idx.npy", ranked_idx)
np.save("feature_importances.npy", rf_scores)

# Optional: save metrics
np.save(
    "validation_metrics.npy",
    accuracy=acc,
    precision=precision,
    recall=recall,
    f1=f1
)